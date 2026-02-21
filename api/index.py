from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import json

app = FastAPI()
client = OpenAI()

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

SYSTEM_PROMPT = """
You are a sentiment analysis engine.
Return ONLY valid JSON in this exact format:

{
  "sentiment": "positive" | "negative" | "neutral",
  "rating": 1-5
}

No extra text. No explanations.
"""

@app.post("/comment", response_model=SentimentResponse)
def analyze_comment(payload: CommentRequest):
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload.comment}
            ]
        )

        raw = response.output_text
        data = json.loads(raw)

        return SentimentResponse(**data)

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model did not return valid JSON")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
