from .model import AMP_model
import requests
from tqdm import tqdm
import os

def download_model_weidht(url, save_path):
    
    try:
        
        response = requests.get(url, stream=True)
        response.raise_for_status()

       
        total_size = int(response.headers.get('Content-Length', 0))
        
    
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

       ）
        progress_bar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {os.path.basename(save_path)} to {os.path.dirname(save_path)} please waiting",
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        
        
        if total_size != 0 and os.path.getsize(save_path) != total_size:
            raise IOError("The downloaded file is incomplete. There might be a network interruption")

        print(f"\nThe file has been saved to：{os.path.abspath(save_path)}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"\nDownload failed (network error): {str(e)}")
        return False
    except IOError as e:
        print(f"\n文件操作失败: {str(e)}")
        
        if os.path.exists(save_path):
            os.remove(save_path)
        return False
    except Exception as e:
        print(f"\nunknown error: {str(e)}")
        return False
