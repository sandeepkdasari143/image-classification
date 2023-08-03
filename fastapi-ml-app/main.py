import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def index():
    return {
        "name": "Sandeep K. Dasari",
        "message": "First route: Hello, World"
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


#uvicorn main:app --reload