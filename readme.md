How to run: 
python -m venv .venv  
pip install -e .  
.venv\Scripts\activate.bat  
python scripts\export_documents.py  
python scripts\export_chunks.py  
docker compose up -d  
python scripts\build_index.py  