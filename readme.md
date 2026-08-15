How to run: 

python -m venv .venv  
pip install -e .  
.venv\Scripts\activate.bat  
python scripts\export_documents.py  
python scripts\export_chunks.py  
docker compose up -d  
python scripts\build_index.py  
python scripts\build_chunks_store.py  
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125  