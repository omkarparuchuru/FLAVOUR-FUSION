from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import requests
import logging
from dotenv import load_dotenv
from typing import Optional, List

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Hugging Face API key
HUGGINGFACE_API_KEY = os.getenv("HGF_API_KEY")
if not HUGGINGFACE_API_KEY:
    raise ValueError("HGF_API_KEY environment variable is not set")

# Hugging Face Model URLs
TEXT_GEN_MODEL = "gpt2"
IMG_GEN_MODEL = "black-forest-labs/FLUX.1-schnell"

HF_API_URL_TEXT = f"https://api-inference.huggingface.co/models/{TEXT_GEN_MODEL}"
HF_API_URL_IMAGE = f"https://api-inference.huggingface.co/models/{IMG_GEN_MODEL}"

HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Initialize FastAPI
app = FastAPI(title="AI Recipe Assistant")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates setup
templates = Jinja2Templates(directory="templates")

# Pydantic Models
class RecipeRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The recipe to generate")
    diet_preference: Optional[str] = Field(None, description="Dietary preference (e.g., vegetarian, vegan)")
    cuisine_type: Optional[str] = Field(None, description="Type of cuisine (e.g., Italian, Mexican)")

class LearningResource(BaseModel):
    title: str
    url: str
    type: str

class RecipeResponse(BaseModel):
    recipe: str
    image_url: str
    learning_resources: List[LearningResource]

# Text Generation Function
def generate_text(prompt: str) -> str:
    """Generates text using Hugging Face API."""
    try:
        logger.info(f"Sending request to Hugging Face: {prompt}")
        response = requests.post(
            HF_API_URL_TEXT,
            headers=HEADERS,
            json={"inputs": prompt}
        )

        logger.info(f"Text Gen Response: {response.status_code} - {response.text}")

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Text generation failed: {response.text}")

        result = response.json()
        return result[0]['generated_text']

    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate text")

# Image Generation Function
def generate_image(prompt: str):
    """Generates an image using Hugging Face API and returns as an image file."""
    try:
        logger.info(f"Generating image for: {prompt}")
        response = requests.post(
            HF_API_URL_IMAGE,
            headers=HEADERS,
            json={"inputs": prompt}
        )

        logger.info(f"Image Gen Response: {response.status_code}")

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Image generation failed: {response.text}")

        return Response(content=response.content, media_type="image/png")

    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate image")

# Generate Recipe
def generate_recipe(query: str, diet_preference: Optional[str] = None, cuisine_type: Optional[str] = None) -> dict:
    """Generates a recipe with text, image, and learning resources."""
    logger.info(f"Generating recipe for: {query}, Diet: {diet_preference}, Cuisine: {cuisine_type}")

    prompt = f"Create a detailed recipe for {query}"
    if diet_preference:
        prompt += f" that is {diet_preference}"
    if cuisine_type:
        prompt += f" in {cuisine_type} style"

    prompt += """
    Format:
    ### Brief Description
    ### Ingredients:
    - List ingredients
    ### Instructions:
    1. Step-by-step instructions
    ### Tips:
    - Cooking tips
    ### Nutritional Information:
    - Calories and health benefits
    """

    # Get text response
    recipe_text = generate_text(prompt)

    # Get image
    image_url = f"/image/{query.replace(' ', '_')}"  # Local API route for image

    # Learning resources
    learning_resources = [
        {
            "title": f"Master the Art of {query}",
            "url": f"https://cooking-school.example.com/learn/{query.lower().replace(' ', '-')}",
            "type": "video"
        },
        {
            "title": f"Tips for Perfect {query}",
            "url": f"https://recipes.example.com/tips/{query.lower().replace(' ', '-')}",
            "type": "article"
        }
    ]

    return {"recipe": recipe_text, "image_url": image_url, "learning_resources": learning_resources}

# API Route for Recipe Generation
@app.post("/recipe", response_model=RecipeResponse)
async def get_recipe(request: RecipeRequest):
    logger.info(f"Received recipe request: {request}")
    try:
        result = generate_recipe(request.query, request.diet_preference, request.cuisine_type)
        return result
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

# API Route for Image Generation
@app.get("/image/{query}")
async def get_image(query: str):
    """Returns an image for the given query."""
    return generate_image(query)

# Root Route (HTML Response)
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Run FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
