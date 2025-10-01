from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from typing import List, Optional
from datetime import date

app = FastAPI(title="My API", description="This is my API")


class User(BaseModel):
    id: int
    name: str
    birth_date: date
    email: Optional[str] = None

# Simulated in-memory database
users_db: List[User] = []

@app.post("/user/", response_model=User, summary="Create a new user")
def create_user(
        id: int = Form(...),
        name: str = Form(...),
        birth_date: date = Form(...),
        email: Optional[str] = Form(None),
):
    patient = User(id=id, name=name, birth_date=birth_date, email=email)
    users_db.append(patient)
    return patient

@app.get("/users/", response_model=List[User], summary="List all users")
async def get_all_users():
    # Just return the global list (don’t re-declare it)
    return users_db