import subprocess, sys
subprocess.run([sys.executable, '-m','unittest','discover','-s',str(FRAMEWORK/'tests'),'-v'],
               cwd=str(FRAMEWORK),check=True)
