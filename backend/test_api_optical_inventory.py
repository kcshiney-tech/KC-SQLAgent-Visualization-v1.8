#!/usr/bin/env python3
# coding: utf-8
"""
fetch_acc_parallel.py

说明:
- 使用 API 返回的最后一条记录的 id 作为下一次 min_id（按用户确认）。
- 请求按序推进（保证 pagination 语义）。
- 使用 ThreadPoolExecutor 并行处理每页返回的数据以提高总体吞吐（网络请求仍顺序）。
- 支持重试、指数退避、去重、结果导出 CSV（也可替换为 DB 写入函数）。
"""

import requests
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import csv
from typing import List, Dict, Any, Optional

# ----------------------------- 配置 -----------------------------
API_URL = "http://10.69.77.238/open-apis/netware/get-acc-info"
ACCESS_TOKEN = "g1erjvg1dmndlkjs3gpmp1483soesjg8"

HEADERS = {
    "Access-Token": ACCESS_TOKEN,
    "Host": "noc.api.sdns.ksyun.com",
    "Content-Type": "application/json",
    "Connection": "keep-alive",
    "Accept": "application/json",
    # 可按需加 User-Agent
}

PAGE_SIZE = 1000
WORKERS = 8                # 并行处理每页的 worker 数
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0      # 秒
SAFETY_LIMIT = 500000      # 最大允许抓取记录数（去重后）
OUTPUT_CSV = "acc_list_output.csv"
LOG_LEVEL = logging.INFO

# ----------------------------- 日志 -----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------------------- 全局并发安全结构 -----------------------------
seen_ids = set()           # 用于去重（仅存 id）——在内存中
seen_lock = Lock()         # 锁保护 seen_ids 与 all_count 写入
all_count = 0
all_count_lock = Lock()

# 存储输出行的临时缓冲（也可以直接写入 DB）
# 为避免内存占用过高，我们在处理函数中直接把每页的记录追加写入 CSV（线程安全写入）
csv_lock = Lock()

# ----------------------------- 工具函数 -----------------------------
def backoff_sleep(attempt: int):
    # 指数退避，带少量抖动
    delay = INITIAL_BACKOFF * (2 ** (attempt - 1))
    jitter = delay * 0.1
    time.sleep(delay + (jitter * (0.5 - time.time() % 1)))  # 简单伪抖动

def safe_post(session: requests.Session, payload: Dict[str, Any], timeout: int = 30) -> requests.Response:
    """带重试的 POST（在请求层面捕获 5xx/连接错误），不捕获 4xx (视为逻辑错误)"""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(API_URL, json=payload, timeout=timeout)
            # 如果是 5xx 或 502，触发重试
            if 500 <= resp.status_code < 600:
                logger.warning(f"请求返回 {resp.status_code}，第 {attempt} 次重试")
                last_exc = RuntimeError(f"HTTP {resp.status_code}")
                backoff_sleep(attempt)
                continue
            return resp
        except requests.RequestException as e:
            last_exc = e
            logger.warning(f"请求异常（第 {attempt} 次）: {e!r}")
            backoff_sleep(attempt)
            continue
    # 全部重试失败，抛出最后一次异常
    logger.error(f"请求重试 {MAX_RETRIES} 次失败")
    raise last_exc

def write_rows_to_csv(rows: List[Dict[str, Any]], csv_file: str = OUTPUT_CSV):
    """线程安全地把一页的 rows 追加写入 CSV（第一次写入会写 header）"""
    if not rows:
        return
    with csv_lock:
        # 追加模式，如果文件不存在写 header
        write_header = False
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                pass
        except FileNotFoundError:
            write_header = True

        # 统一列集合（取所有 key 的并集，保证列齐）
        keys = set()
        for r in rows:
            keys.update(r.keys())
        keys = sorted(keys)

        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                writer.writeheader()
            for r in rows:
                # 将非字符串化为 JSON 字符串以保留复杂字段
                row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in r.items()}
                writer.writerow(row)

# ----------------------------- 页面处理函数（并行执行） -----------------------------
def process_page_and_persist(page_items: List[Dict[str, Any]]) -> int:
    """
    对单页数据做去重并写入（CSV/DB）。返回本页新增的记录数。
    这个函数会被 ThreadPoolExecutor 并行执行以提高吞吐。
    """
    global all_count
    new_items = []
    added = 0
    # 局部集合以减少锁竞争
    local_new_ids = []
    for item in page_items:
        iid = item.get("id")
        if iid is None:
            # 如果没有 id，可以选择跳过或保存。这里跳过并记录日志
            logger.debug("跳过无 id 的记录")
            continue
        # 使用锁检查并添加到 seen_ids
        with seen_lock:
            if iid in seen_ids:
                continue
            seen_ids.add(iid)
        local_new_ids.append(iid)
        new_items.append(item)

    added = len(new_items)
    if added > 0:
        # 持久化：这里示范写 CSV，也可以替换为写数据库函数
        write_rows_to_csv(new_items, OUTPUT_CSV)
        with all_count_lock:
            all_count += added

    return added

# ----------------------------- 主抓取函数 -----------------------------
def fetch_all(page_size: int = PAGE_SIZE, workers: int = WORKERS, safety_limit: int = SAFETY_LIMIT, output_csv: Optional[str] = OUTPUT_CSV):
    """
    主函数：顺序请求分页 (min_id = last returned item's id)，并把每页提交给线程池并行处理。
    """
    global all_count
    session = requests.Session()
    session.trust_env = False
    session.headers.update(HEADERS)

    # 初始化/清空输出 CSV 文件（覆盖）
    if output_csv:
        # 创建空文件（会在并行写入时追加 header）
        with open(output_csv, "w", encoding="utf-8") as f:
            pass

    min_id = 0
    total_reported = None
    page_no = 0
    # 记录失败页信息
    failed_pages = []

    # 线程池用于并行处理每页
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []  # 保存正在处理的 Future

        while True:
            page_no += 1
            payload = {
                "idc": "",
                "producer": "",
                "container": "业务部门网络零件",
                "class": "光模块",
                "status": ["在线", "流程处理中"],
                "page_size": page_size,
                "min_id": min_id
            }

            try:
                resp = safe_post(session, payload, timeout=30)
            except Exception as e:
                logger.exception(f"第 {page_no} 页请求失败，退出抓取: {e!r}")
                failed_pages.append((page_no, str(e)))
                break

            # 若非200，记录并根据需要退出
            if resp.status_code != 200:
                logger.error(f"第 {page_no} 页返回 HTTP {resp.status_code}：{resp.text[:200]}")
                failed_pages.append((page_no, f"HTTP {resp.status_code}"))
                break

            try:
                result = resp.json()
            except Exception as e:
                logger.exception(f"第 {page_no} 页 JSON 解析失败，响应头: {resp.headers}")
                failed_pages.append((page_no, "json_parse_error"))
                break

            # 解析 total（后续页可能变化）
            page_total = result.get("data", {}).get("total")
            if total_reported is None:
                total_reported = page_total
                logger.info(f"首次返回 total = {total_reported}")
            else:
                if page_total != total_reported:
                    logger.info(f"第 {page_no} 页返回 total = {page_total}（变化，之前={total_reported}）")
                    total_reported = page_total

            acc_list = result.get("data", {}).get("acc_list", []) or []
            if not acc_list:
                logger.info(f"第 {page_no} 页无数据，结束")
                break

            # 按你确认的规则：使用返回 json 的最后一条记录的 id 作为下一次 min_id
            last_rec = acc_list[-1]
            last_id = last_rec.get("id")
            if last_id is None:
                logger.error(f"第 {page_no} 页最后一条记录无 id，停止抓取以避免死循环")
                failed_pages.append((page_no, "last_id_missing"))
                break

            # 把当前页的 acc_list 提交给线程池并行处理
            future = pool.submit(process_page_and_persist, acc_list)
            futures.append((page_no, future))

            # 更新 min_id（按你确认的逻辑直接设置为返回的最后一条 id）
            min_id = last_id

            # 进度打印：统计已去重计数（注意处理可能仍在进行中）
            with all_count_lock:
                current_count = all_count
            logger.info(f"已请求页 {page_no}，返回 {len(acc_list)} 条，累计已去重 {current_count} 条 (reported total={page_total})")

            # 安全上限检查（去重后）
            if current_count >= safety_limit:
                logger.warning(f"到达安全上限 {safety_limit} 条，停止抓取")
                break

            # 小睡以防某些网关限制频率（可根据需要调整或移除）
            # time.sleep(0.01)

        # 等待所有提交的处理任务完成（或超时/取消）
        logger.info("等待并行处理任务完成...")
        for page_idx, fut in futures:
            try:
                added = fut.result(timeout=300)  # 可根据需要调整超时时间
                logger.debug(f"page {page_idx} 处理完成，新增 {added} 条")
            except Exception as e:
                logger.exception(f"page {page_idx} 处理异常: {e!r}")
                failed_pages.append((page_idx, f"process_error:{e!r}"))

    # 最终统计
    with all_count_lock:
        final_count = all_count
    logger.info(f"抓取结束。去重后总记录数: {final_count}")
    if total_reported is not None:
        logger.info(f"最后一次 page_total 报告为: {total_reported}")
    if failed_pages:
        logger.warning(f"部分页失败/异常: {failed_pages}")

    return {
        "total_fetched_unique": final_count,
        "last_reported_total": total_reported,
        "failed_pages": failed_pages
    }

# ----------------------------- 主运行示例 -----------------------------
if __name__ == "__main__":
    start = time.time()
    result = fetch_all(page_size=PAGE_SIZE, workers=WORKERS, safety_limit=SAFETY_LIMIT, output_csv=OUTPUT_CSV)
    elapsed = time.time() - start
    logger.info(f"结果: {result}")
    logger.info(f"耗时: {elapsed:.1f}s")
