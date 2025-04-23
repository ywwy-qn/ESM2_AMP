from .model import AMP_model
import requests
from tqdm import tqdm
import os

def download_model_weidht(url, save_path):
    """
    带进度条的下载函数
    
    参数：
    url (str): 要下载的文件URL
    save_path (str): 文件保存的完整路径
    
    返回：
    bool: 下载是否成功的状态 (True/False)
    """
    try:
        # 发起带流式传输的GET请求
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查HTTP错误

        # 获取文件总大小（字节）
        total_size = int(response.headers.get('Content-Length', 0))
        
        # 创建目标目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 初始化进度条（显示下载路径信息）
        progress_bar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {os.path.basename(save_path)} to {os.path.dirname(save_path)} please waiting",
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        # 分块写入文件
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # 过滤保持活跃状态的空chunk
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        
        # 验证下载完整性（如果服务器提供了Content-Length）
        if total_size != 0 and os.path.getsize(save_path) != total_size:
            raise IOError("下载文件不完整，可能网络中断")

        print(f"\n文件已保存至：{os.path.abspath(save_path)}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"\n下载失败（网络错误）: {str(e)}")
        return False
    except IOError as e:
        print(f"\n文件操作失败: {str(e)}")
        # 删除可能不完整的文件
        if os.path.exists(save_path):
            os.remove(save_path)
        return False
    except Exception as e:
        print(f"\n未知错误: {str(e)}")
        return False