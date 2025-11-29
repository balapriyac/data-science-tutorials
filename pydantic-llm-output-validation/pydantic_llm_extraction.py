"""
Pydantic with LLM APIs - Complete Implementation
Demonstrates structured data extraction using OpenAI, LangChain, and LlamaIndex
"""

import os
import json
from typing import List, Optional
from enum import Enum
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =============================================================================
# PART 1: DIRECT OPENAI API USAGE
# =============================================================================

class BookSummary(BaseModel):
    title: str
    author: str
    genre: str
    key_themes: List[str]
    main_characters: List[str]
    brief_summary: str
    recommended_for: List[str]


def extract_book_info(text: str) -> BookSummary:
    """Extract structured book information from unstructured text."""
    
    prompt = f"""
    Extract book information from the following text and return it as JSON.
    
    Required format:
    {{
        "title": "book title",
        "author": "author name",
        "genre": "genre",
        "key_themes": ["theme1", "theme2"],
        "main_characters": ["character1", "character2"],
        "brief_summary": "summary in 2-3 sentences",
        "recommended_for": ["audience1", "audience2"]
    }}
    
    Text: {text}
    
    Return ONLY the JSON, no additional text.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts structured data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    llm_output = response.choices[0].message.content
    data = json.loads(llm_output)
    return BookSummary(**data)


# =============================================================================
# PART 2: LANGCHAIN WITH PYDANTIC
# =============================================================================

from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate


class PriceRange(str, Enum):
    BUDGET = "$"
    MODERATE = "$$"
    EXPENSIVE = "$$$"
    LUXURY = "$$$$"


class Restaurant(BaseModel):
    """Information about a restaurant."""
    name: str = Field(description="The name of the restaurant")
    cuisine: str = Field(description="Type of cuisine served")
    price_range: str = Field(description="Price range: $, $$, $$$, or $$$$")
    rating: Optional[float] = Field(default=None, description="Rating out of 5.0")
    specialties: List[str] = Field(description="Signature dishes or specialties")


def extract_restaurant_with_parser(text: str) -> Restaurant:
    """Extract restaurant info using LangChain's PydanticOutputParser."""
    
    parser = PydanticOutputParser(pydantic_object=Restaurant)
    
    prompt = PromptTemplate(
        template="Extract restaurant information from the following text.\n{format_instructions}\n{text}\n",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | parser
    result = chain.invoke({"text": text})
    return result


def extract_restaurant_structured(text: str) -> Restaurant:
    """Extract restaurant info using with_structured_output."""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(Restaurant)
    
    prompt = PromptTemplate.from_template(
        "Extract restaurant information from the following text:\n\n{text}"
    )
    
    chain = prompt | structured_llm
    result = chain.invoke({"text": text})
    return result


# =============================================================================
# PART 3: LLAMAINDEX WITH PYDANTIC
# =============================================================================

from llama_index.core.program import LLMTextCompletionProgram, FunctionCallingProgram
from llama_index.core.output_parsers import PydanticOutputParser as LlamaIndexPydanticOutputParser
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI


class Product(BaseModel):
    """Information about a product."""
    name: str = Field(description="Product name")
    brand: str = Field(description="Brand or manufacturer")
    category: str = Field(description="Product category")
    price: float = Field(description="Price in USD")
    features: List[str] = Field(description="Key features")
    rating: Optional[float] = Field(default=None, description="Customer rating out of 5")


def extract_product_simple(text: str) -> Product:
    """Extract product info using LlamaIndex's simple approach."""
    
    prompt_template_str = """
    Extract product information from the following text and structure it properly:
    
    {text}
    """
    
    program = LLMTextCompletionProgram.from_defaults(
        output_cls=Product,
        prompt_template_str=prompt_template_str,
        verbose=False
    )
    
    result = program(text=text)
    return result


def extract_product_function_calling(text: str) -> Product:
    """Extract product info using function calling."""
    
    prompt_template_str = """
    Extract product information from the following text:
    
    {text}
    """
    
    program = FunctionCallingProgram.from_defaults(
        output_cls=Product,
        prompt_template_str=prompt_template_str,
        verbose=False
    )
    
    result = program(text=text)
    return result


def extract_product_with_parser(text: str) -> Product:
    """Extract product info using explicit parser."""
    
    prompt_template_str = """
    Extract product information from the following text:
    
    {text}
    
    {format_instructions}
    """
    
    llm = LlamaIndexOpenAI(model="gpt-4o-mini", temperature=0)
    
    program = LLMTextCompletionProgram.from_defaults(
        output_parser=LlamaIndexPydanticOutputParser(output_cls=Product),
        prompt_template_str=prompt_template_str,
        llm=llm,
        verbose=False
    )
    
    result = program(text=text)
    return result


# =============================================================================
# EXAMPLE USAGE AND TESTS
# =============================================================================

def main():
    """Run example extractions with all methods."""
    
    print("=" * 80)
    print("PYDANTIC + LLM EXTRACTION EXAMPLES")
    print("=" * 80)
    
    # Example 1: Book info extraction with OpenAI
    print("\n1. BOOK EXTRACTION (Direct OpenAI API)")
    print("-" * 80)
    book_text = """
    'The Midnight Library' by Matt Haig is a contemporary fiction novel that explores 
    themes of regret, mental health, and the infinite possibilities of life. The story 
    follows Nora Seed, a woman who finds herself in a library between life and death, 
    where each book represents a different life she could have lived. Through her journey, 
    she encounters various versions of herself and must decide what truly makes a life worth living.
    The book resonates with readers dealing with depression, anxiety, or life transitions.
    """
    
    try:
        book_info = extract_book_info(book_text)
        print(f"Title: {book_info.title}")
        print(f"Author: {book_info.author}")
        print(f"Genre: {book_info.genre}")
        print(f"Themes: {', '.join(book_info.key_themes)}")
        print(f"Characters: {', '.join(book_info.main_characters)}")
        print(f"Summary: {book_info.brief_summary}")
    except Exception as e:
        print(f"Error extracting book info: {e}")
    
    # Example 2: Restaurant extraction with LangChain
    print("\n\n2. RESTAURANT EXTRACTION (LangChain with_structured_output)")
    print("-" * 80)
    restaurant_text = """
    Mama's Italian Kitchen is a cozy family-owned restaurant serving authentic 
    Italian cuisine. Rated 4.5 stars, it's known for its homemade pasta and 
    wood-fired pizzas. Prices are moderate ($$), and their signature dishes 
    include lasagna bolognese and tiramisu.
    """
    
    try:
        restaurant_info = extract_restaurant_structured(restaurant_text)
        print(f"Restaurant: {restaurant_info.name}")
        print(f"Cuisine: {restaurant_info.cuisine}")
        print(f"Price Range: {restaurant_info.price_range}")
        print(f"Rating: {restaurant_info.rating}/5.0")
        print(f"Specialties: {', '.join(restaurant_info.specialties)}")
    except Exception as e:
        print(f"Error extracting restaurant info: {e}")
    
    # Example 3: Product extraction with LlamaIndex
    print("\n\n3. PRODUCT EXTRACTION (LlamaIndex FunctionCallingProgram)")
    print("-" * 80)
    product_text = """
    The Sony WH-1000XM5 wireless headphones feature industry-leading noise cancellation,
    exceptional sound quality, and up to 30 hours of battery life. Priced at $399.99,
    these premium headphones include Adaptive Sound Control, multipoint connection,
    and speak-to-chat technology. Customers rate them 4.7 out of 5 stars.
    """
    
    try:
        product_info = extract_product_function_calling(product_text)
        print(f"Product: {product_info.name}")
        print(f"Brand: {product_info.brand}")
        print(f"Category: {product_info.category}")
        print(f"Price: ${product_info.price}")
        print(f"Rating: {product_info.rating}/5.0")
        print(f"Features: {', '.join(product_info.features)}")
    except Exception as e:
        print(f"Error extracting product info: {e}")
    
    print("\n" + "=" * 80)
    print("All extractions completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()


