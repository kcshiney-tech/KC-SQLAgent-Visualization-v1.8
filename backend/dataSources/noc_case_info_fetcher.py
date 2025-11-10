import json
import requests

# 常量定义
CASE_INFO_URL = "https://noc.ksyun.com/interface/get-case-info"
ACCESS_TOKEN = "g1erjvg1dmndlkjs3gpmp1483soesjg8"

def fetch_case_info(case_id):
    """
    通过case_id获取工单详细信息
    
    Args:
        case_id (str): 工单ID
        
    Returns:
        dict: 提取的工单信息
    """
    # 准备请求数据
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    data = {
        "access_token": ACCESS_TOKEN,
        "case_id": case_id
    }
    
    # 发送POST请求获取数据
    response = requests.post(CASE_INFO_URL, headers=headers, json=data)
    
    # 解析JSON（处理双重转义的情况）
    # 首先解析外层的JSON字符串
    outer_data = response.json()
    # 然后解析内层的JSON字符串
    data = json.loads(outer_data)
    
    # 创建一个字典来存储所有提取的信息
    extracted_info = {}
    
    # 提取基本信息
    extracted_info['case_id'] = data.get('data', {}).get('case_id', '')
    extracted_info['status'] = data.get('data', {}).get('status', '')
    extracted_info['title'] = data.get('data', {}).get('title', '')
    extracted_info['fault_type'] = data.get('data', {}).get('fault_type', '')
    extracted_info['fault_msg'] = data.get('data', {}).get('fault_msg', '')
    extracted_info['op_time'] = data.get('data', {}).get('op_time', '')
    extracted_info['create_time'] = data.get('data', {}).get('create_time', '')
    extracted_info['last_update_time'] = data.get('data', {}).get('last_update_time', '')
    extracted_info['creator'] = data.get('data', {}).get('creator', '')
    extracted_info['idc'] = data.get('data', {}).get('idc', '')
    
    # 提取额外信息
    extracted_info['status_code'] = data.get('data', {}).get('status_code', '')
    extracted_info['process_id'] = data.get('data', {}).get('process_id', '')
    extracted_info['list_id'] = data.get('data', {}).get('list_id', '')
    extracted_info['list_type'] = data.get('data', {}).get('list_type', '')
    extracted_info['table_name'] = data.get('data', {}).get('table_name', '')
    extracted_info['operate_name'] = data.get('data', {}).get('operate_name', '')
    extracted_info['emergency'] = data.get('data', {}).get('emergency', '')
    extracted_info['auto_complete'] = data.get('data', {}).get('auto_complete', '')
    extracted_info['auto_create_outsource'] = data.get('data', {}).get('auto_create_outsource', '')
    extracted_info['module_adminer'] = data.get('data', {}).get('module_adminer', '')
    extracted_info['outsource'] = data.get('data', {}).get('outsource', [])
    
    # 解码Unicode转义序列
    if isinstance(extracted_info['fault_type'], str):
        extracted_info['fault_type'] = json.loads(f'"{extracted_info["fault_type"]}"')
    if isinstance(extracted_info['fault_msg'], str):
        extracted_info['fault_msg'] = json.loads(f'"{extracted_info["fault_msg"]}"')
    if isinstance(extracted_info['title'], str):
        extracted_info['title'] = json.loads(f'"{extracted_info["title"]}"')
    
    # 提取SN和clazz信息
    sn_clazz_list = []
    if 'data' in data and 'netware_list' in data['data']:
        netware_list = data['data']['netware_list']
        for item in netware_list:
            sn = item.get('sn', '')
            clazz = item.get('clazz', '')
            replace_sn = item.get('replace_sn', '')
            is_fault = item.get('is_fault', '')
            id = item.get('id', '')
            
            # 解码clazz中的Unicode转义序列
            if isinstance(clazz, str):
                clazz = json.loads(f'"{clazz}"')
                
            sn_clazz_list.append({
                'sn': sn,
                'clazz': clazz,
                'replace_sn': replace_sn,
                'is_fault': is_fault,
                'id': id
            })
    extracted_info['netware_list'] = sn_clazz_list
    
    # 提取list_device信息
    list_device = data.get('data', {}).get('list_device', '')
    if isinstance(list_device, str):
        list_device = json.loads(f'"{list_device}"')
    extracted_info['list_device'] = list_device
    
    # 提取desc信息
    desc = data.get('data', {}).get('desc', '')
    if isinstance(desc, str):
        try:
            desc = json.loads(f'"{desc}"')
        except json.JSONDecodeError:
            # 如果解码失败，保留原始字符串
            pass
    extracted_info['desc'] = desc
    
    return extracted_info

def get_sn_and_types(case_id):
    """
    通过case_id获取SN和类型信息
    
    Args:
        case_id (str): 工单ID
        
    Returns:
        list: 包含SN和类型信息的列表
    """
    # 获取完整工单信息
    case_info = fetch_case_info(case_id)
    
    # 返回网络设备列表中的SN和类型信息
    return case_info.get('netware_list', [])

# 示例用法
if __name__ == "__main__":
    # 示例：获取工单信息
    case_id = "1088464"
    info = fetch_case_info(case_id)
    
    print("="*60)
    print("工单基本信息:")
    print("="*60)
    print(f"工单ID: {info['case_id']}")
    print(f"状态: {info['status']}")
    print(f"状态码: {info['status_code']}")
    print(f"标题: {info['title']}")
    print(f"故障类型: {info['fault_type']}")
    print(f"故障消息: {info['fault_msg']}")
    print(f"操作时间: {info['op_time']}")
    print(f"创建时间: {info['create_time']}")
    print(f"最后更新时间: {info['last_update_time']}")
    print(f"创建者: {info['creator']}")
    print(f"机房: {info['idc']}")
    print(f"紧急程度: {info['emergency']}")
    print(f"自动完成: {info['auto_complete']}")
    print(f"自动创建外包: {info['auto_create_outsource']}")
    print(f"模块管理员: {info['module_adminer']}")
    print(f"外包工单: {info['outsource'][0]}")
    
    print("\n" + "="*60)
    print("流程相关信息:")
    print("="*60)
    print(f"流程ID: {info['process_id']}")
    print(f"列表ID: {info['list_id']}")
    print(f"列表类型: {info['list_type']}")
    print(f"表名: {info['table_name']}")
    print(f"操作名称: {info['operate_name']}")
    
    print("\n" + "="*60)
    print("网络设备列表:")
    print("="*60)
    for item in info['netware_list']:
        print(f"  SN: {item['sn']}, 类型: {item['clazz']}, 替换SN: {item['replace_sn']}, 故障: {item['is_fault']}, ID: {item['id']}")
    
    print("\n" + "="*60)
    print("设备列表描述:")
    print("="*60)
    print(f"{info['list_device']}")
    
    print("\n" + "="*60)
    print("详细描述:")
    print("="*60)
    print(f"{info['desc']}")
    
    print("\n" + "="*60)
    
    # 示例：仅获取SN和类型信息
    sn_types = get_sn_and_types(case_id)
    print("\nSN和类型信息:")
    for item in sn_types:
        print(f"  SN: {item['sn']}, 类型: {item['clazz']}")
