import zipfile
import tomllib

def extract_osu_file(osz_path):
    with zipfile.ZipFile(osz_path, "r") as z:
            file_list = z.namelist()
            
            target_files = [f for f in file_list if f.lower().endswith(".osu")]
            
            for file in target_files:
                with z.open(file) as f:
                    content = f.read().decode('utf-8', errors='replace')
                    return content
    return None

def parse_osu_file(osu_content):
    if not osu_content:
        return None
        
    sections = {}
    current_section = None
    
    for line in osu_content.splitlines():
        line = line.strip()
        
        if not line:
            continue
            
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1]  # Remove brackets
            sections[current_section] = []
            continue
            
        if current_section:
            sections[current_section].append(line)
    
    return sections    


def parse_osz_file(osz_path):
    content = extract_osu_file(osz_path)
    if content is None:
        return None
    
    for k, v in parse_osu_file(content).items():
        print(k,len(v))
        