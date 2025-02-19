from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Value(BaseModel):
    value: int

# Welcome user
@app.get("/")
def welcome():
    return {"Hello": "Welcome!"}

