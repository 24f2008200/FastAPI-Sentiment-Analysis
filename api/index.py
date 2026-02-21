from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI()

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

@app.post("/comment", response_model=SentimentResponse)
def analyze_comment(payload: CommentRequest):
    try:
        response = client.responses.parse(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": "Analyze sentiment and return JSON only."},
                {"role": "user", "content": payload.comment}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_analysis",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"]
                            },
                            "rating": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": ["sentiment", "rating"],
                        "additionalProperties": False
                    }
                }
            }
        )
        return response.output_parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
