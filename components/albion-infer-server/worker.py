from pydantic import BaseModel

data = {}
class Log(BaseModel):
   id: int
   status: str

@app.post("/status")
def add_book(log: Log):
   data[log.id] = log.status
   return data

@app.get("/status/{id}")
def get_books(id:int):
   return data.get(id)

