# api-modelo
1. Entorno virtual
```
python -m venv env
```
2. 
```
source env/bin/activate  
```
3. Instalar dependencias:
```
pip install fastapi uvicorn fasttext
```
4. Levantar api:
```
uvicorn main:app --reload
```
5. Probar api:
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "Blanqueamiento dentario con Laser (ambas arcadas - estético)"}'
```