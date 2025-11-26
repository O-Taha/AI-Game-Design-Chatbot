import subprocess
from pathlib import Path

def safe_git_pull(repo_path: str = './mechanics') -> bool:
    """
    Tente de faire un git pull dans repo_path.
    Retourne True si pull réussi (ou déjà à jour), False sinon.
    """
    result = subprocess.run(
        ['git', '-C', repo_path, 'pull'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        print(f"[WARNING] git pull failed:\n{result.stderr.decode()}")
        return False
    return True

def get_latest_commit_hash(repo_path: str = './mechanics') -> str:
    result = subprocess.run(
        ['git', '-C', repo_path, 'rev-parse', 'HEAD'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.decode('utf-8').strip()

def is_new_commit(repo_path: str = './mechanics') -> bool:
    """
    Effectue un git pull, puis compare le HEAD avant/après.
    Retourne True si le commit a changé, False sinon.
    """  
    last_commit_file = Path('mechanics/.last_commit')
    latest_commit = get_latest_commit_hash(repo_path)
    
    if last_commit_file.exists():
        stored_commit = last_commit_file.read_text().strip()
        if stored_commit == latest_commit:
            return False
    
    last_commit_file.write_text(latest_commit)

    return True
