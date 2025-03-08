import random
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import pandas as pd
import spacy

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model for more advanced text manipulation
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")



# Specific fruits and plant parts as requested
FRUITS = [
    # Common fruits
    "apple", "banana", "orange", "strawberry", "grape", "watermelon", "mango", 
    "pineapple", "peach", "plum", "pear", "cherry", "blueberry", "raspberry",
    "blackberry", "kiwi", "lemon", "lime", "grapefruit", "avocado", "coconut",
    "fig", "pomegranate", "papaya", "guava", "passion fruit", "dragonfruit",
    
    # Vegetables often treated as fruits
    "tomato", "cucumber", "bell pepper", "eggplant", "zucchini", "pumpkin",
    "squash", "okra"
]
# Expanded vegetables list
VEGETABLES = [
    # Root vegetables
    "potato", "carrot", "onion", "garlic", "ginger", "radish", "turnip",
    "beetroot", "sweet potato", "yam", "cassava", "parsnip", "rutabaga",
    
    # Leafy greens
    "lettuce", "spinach", "kale", "cabbage", "arugula", "collard greens",
    "swiss chard", "mustard greens", "bok choy", "watercress",
    
    # Other vegetables
    "broccoli", "cauliflower", "brussels sprout", "celery", "asparagus",
    "artichoke", "leek", "green bean", "mushroom", "corn", "pea"
]

# Expanded plant parts
PLANT_PARTS = [
    "leaf", "stem", "root", "flower", "bud", "shoot", "bulb", "tuber",
    "rhizome", "sprout", "seedling"
]


# Expanded conditions for fruits and vegetables
CONDITIONS = [
    # Ripeness conditions
    "ripe", "overripe", "unripe", "green", "mature", "immature",
    
    # Colors
    "yellow", "red", "purple", "orange", "brown", "pale", "dark",
    "mottled", "striped", "speckled", "blotchy",
    
    # Texture conditions
    "soft", "hard", "mushy", "firm", "tender", "crisp", "juicy", "dry",
    "rubbery", "fibrous", "woody", "mealy", "grainy", "gritty",
    
    # Deterioration conditions
    "bruised", "damaged", "cracked", "split", "wrinkled", "shriveled",
    "wilted", "dehydrated", "dented", "scarred", "pitted", "blemished",
    
    # Decay conditions
    "rotten", "moldy", "fermented", "slimy", "oozing", "sour",
    "fuzzy", "spotted", "discolored", "bug-infested", "wormy",
    
    # Storage conditions
    "refrigerated", "frozen", "thawed", "chilled", "warm", "room temperature",
    "sun-dried", "dehydrated", "preserved", "waxed", "unwaxed", "sealed",
    
    # Preparation states
    "washed", "unwashed", "peeled", "unpeeled", "cut", "sliced", "whole",
    "chopped", "diced", "grated", "pureed", "juiced",
    
    # Growth/Production conditions
    "organic", "conventional", "pesticide-free", "GMO", "non-GMO",
    "locally grown", "imported", "greenhouse-grown", "hydroponically grown"
]

# Expanded plant disease and pest conditions
PLANT_DISEASES = [
    # Common plant diseases
    "powdery mildew", "downy mildew", "leaf spot", "leaf rust", "leaf curl",
    "blight", "root rot", "stem rot", "bacterial wilt", "fusarium wilt",
    "anthracnose", "mosaic virus", "black spot", "canker", "scab",
    "sooty mold", "fire blight", "crown gall", "clubroot", "damping off",
    
    # Pest damage
    "aphid infestation", "whitefly damage", "spider mite damage",
    "scale insect", "thrips damage", "caterpillar damage", "mealybug", 
    "leaf miner trails", "slug damage", "snail damage", "beetle damage",
    "weevil damage", "fruit fly infestation", "nematode damage"
]

# Define personas with their characteristics
PERSONAS = {
    "farmer": {
        "vocabulary": ["crop", "yield", "harvest", "fertilizer", "pesticide", "irrigation", 
                       "soil", "seedling", "growth", "nutrient", "spray", "field", "mulch", 
                       "organic", "compost", "season", "cultivate", "transplant", "propagate"],
        "complexity": "medium",
        "interests": ["plant_health", "pest_management", "harvest_timing", "crop_yield", 
                      "soil_conditions", "weather_impact", "storage", "market_readiness"],
        "question_style": ["direct", "practical", "specific", "solution-oriented"],
        "regional_terms": {
            "US_midwest": ["field", "plot", "acre"],
            "US_south": ["patch", "row crop", "truck farm"],
            "UK": ["allotment", "polytunnel", "smallholding"],
            "india": ["khet", "kisaan", "rabi", "kharif"]
        },
        "examples": [
            "My tomato plants got yellow leaves, what's wrong?",
            "When's the best time to harvest these potatoes?",
            "Will this frost damage my strawberry crop?",
            "How do I stop bugs eating my cabbage?"
        ]
    },
    
    "agricultural_engineer": {
        "vocabulary": ["irrigation system", "crop management", "optimization", "yield analysis", 
                      "efficiency", "mechanization", "drainage", "precision agriculture", 
                      "hydroponics", "nutrient management", "water usage", "automation"],
        "complexity": "high",
        "interests": ["system_design", "efficiency", "technology_integration", "resource_management",
                      "scalability", "data_analysis", "sustainability", "optimization"],
        "question_style": ["analytical", "technical", "systemic", "quantitative"],
        "examples": [
            "What's the optimal irrigation frequency for these pepper plants in sandy loam soil?",
            "How does the nutrient content change during storage of these onions?",
            "What's the relationship between calcium deficiency and blossom end rot in tomatoes?",
            "Can you explain the ideal storage conditions to maximize shelf life for these mangoes?"
        ]
    },
    
    "agricultural_trader": {
        "vocabulary": ["market", "price", "quality", "grade", "export", "import", "supply chain", 
                      "shipping", "wholesale", "retail", "certification", "standards", "shelf life"],
        "complexity": "medium-high",
        "interests": ["quality_assessment", "market_value", "shelf_life", "transportation", 
                     "packaging", "grading", "trade_regulations", "certification"],
        "question_style": ["commercial", "evaluative", "comparative", "market-focused"],
        "examples": [
            "Are these apples export quality?",
            "Will these strawberries last during a 3-day shipment?",
            "Does this carrot quality meet supermarket standards?",
            "How can I tell if this batch of oranges is premium grade?"
        ]
    },
    
    "environmental_engineer": {
        "vocabulary": ["sustainable", "ecological", "runoff", "contamination", "biodiversity", 
                      "carbon footprint", "organic methods", "integrated pest management", 
                      "water conservation", "soil health", "ecosystem"],
        "complexity": "high",
        "interests": ["sustainability", "ecological_impact", "water_management", "soil_health", 
                     "biodiversity", "organic_practices", "climate_adaptation"],
        "question_style": ["systemic", "environmentally-focused", "long-term", "impact-oriented"],
        "examples": [
            "Are these pesticide residue levels normal on this lettuce?",
            "What sustainable alternatives exist for managing this tomato blight?",
            "How does this organic growing method affect soil microbial activity?",
            "What's the water footprint of growing these cucumbers?"
        ]
    },
    
    "inspector": {
        "vocabulary": ["compliance", "standard", "regulation", "inspection", "certification", 
                      "safety", "protocol", "documentation", "traceability", "audit", 
                      "violation", "assessment", "threshold"],
        "complexity": "high",
        "interests": ["regulatory_compliance", "quality_control", "safety_standards", 
                     "certification_requirements", "contamination", "documentation"],
        "question_style": ["methodical", "standards-based", "detailed", "regulatory"],
        "examples": [
            "Does this tomato damage meet exclusion criteria for grade A?",
            "Is this level of pesticide residue within acceptable limits?",
            "Do these apples meet organic certification requirements?",
            "Are these storage practices compliant with food safety regulations?"
        ]
    },
    
    "quality_controller": {
        "vocabulary": ["quality check", "specification", "defect", "uniformity", "consistency", 
                      "sampling", "batch", "sensory evaluation", "attribute", "standard operating procedure"],
        "complexity": "medium-high",
        "interests": ["quality_assessment", "defect_identification", "grading", "uniformity", 
                      "sensory_properties", "shelf_life", "packaging_integrity"],
        "question_style": ["detail-oriented", "standardized", "procedural", "comparative"],
        "examples": [
            "Are these brown spots on lettuce within acceptable quality limits?",
            "Does the ripeness of these bananas meet our shipping specifications?",
            "Is this color variation in bell peppers normal or a defect?",
            "How can I test firmness consistency across this batch of avocados?"
        ]
    },
    
    "final_consumer": {
        "vocabulary": ["fresh", "tasty", "ripe", "eat", "cook", "recipe", "store", 
                      "healthy", "organic", "GMO", "price", "buy", "spoiled", "bad"],
        "complexity": "low",
        "interests": ["edibility", "taste", "nutritional_value", "storage", "preparation", 
                     "ripeness", "selection", "safety", "organic_vs_conventional"],
        "question_style": ["simple", "practical", "usage-focused", "immediate"],
        "examples": [
            "Is this apple still good to eat?",
            "How do I know when this avocado is ripe?",
            "Can I eat strawberries with these white spots?",
            "Should I keep tomatoes in the fridge?"
        ]
    },
    
    "scientist": {
        "vocabulary": ["cultivar", "variety", "genotype", "phenotype", "physiological", 
                      "metabolic", "photosynthetic", "morphology", "taxonomy", "hybrid", 
                      "genetic", "biochemical", "pathogen"],
        "complexity": "very high",
        "interests": ["genetic_factors", "biochemical_composition", "physiological_processes", 
                     "pathogen_identification", "variety_characteristics", "experimental_data"],
        "question_style": ["analytical", "precise", "technical", "academic", "research-oriented"],
        "examples": [
            "What physiological factors affect anthocyanin development in these berries?",
            "How does ethylene production correlate with ripening stages in these mangoes?",
            "Can you identify the specific pathogen causing these lesions on tomato leaves?",
            "What cultivar of cucumber is this based on morphological characteristics?"
        ]
    },
    
    "biologist": {
        "vocabulary": ["organism", "species", "cellular", "metabolic", "biological", 
                      "enzyme", "respiration", "photosynthesis", "chlorophyll", 
                      "vascular", "mitochondria", "cytoplasm"],
        "complexity": "high",
        "interests": ["biological_processes", "cellular_structure", "species_identification", 
                     "metabolic_pathways", "developmental_biology", "environmental_adaptation"],
        "question_style": ["scientific", "descriptive", "process-oriented", "classificatory"],
        "examples": [
            "What's causing this unusual cellular structure in these strawberry leaves?",
            "How does cold storage affect the metabolic rate in these apples?",
            "What type of fungal species is growing on this fruit?",
            "Can you explain the physiological changes during ripening in these bananas?"
        ]
    },
    
    "naturalist": {
        "vocabulary": ["native", "wild", "species", "habitat", "ecosystem", "biodiversity", 
                      "pollinator", "foraging", "seasonal", "indigenous", "edible", "medicinal"],
        "complexity": "medium-high",
        "interests": ["species_identification", "ecological_relationships", "native_varieties", 
                     "seasonality", "traditional_uses", "sustainable_harvesting", "wild_edibles"],
        "question_style": ["observational", "ecological", "holistic", "naturalistic", "curious"],
        "examples": [
            "Is this wild berry safe to eat?",
            "What variety of wild apple is this?",
            "How can I tell if these mushrooms are edible?",
            "Is this plant native to this region?"
        ]
    }
}

# Templates
# Intent templates with complexity levels
INTENT_TEMPLATES = {
    "fruit_quality": {
        "low": [
            "Is this {produce} good?",
            "Can I eat this {produce}?", 
            "Is this {produce} bad?",
            "Is this {produce} safe to eat?",
            "Does this {produce} look okay?",
            "Is it okay if my {produce} is {condition}?",
            "Should I throw away this {produce}?",
            "My {produce} looks {condition}, is that normal?"
        ],
        "medium": [
            "Is this {condition} {produce} still edible?",
            "Does this {produce} have acceptable quality for consumption?",
            "What can you tell me about the quality of this {condition} {produce}?",
            "Is it safe to consume {produce} that shows signs of being {condition}?",
            "How can I determine if this {condition} {produce} is still good?"
        ],
        "high": [
            "Can you assess the quality parameters of this {condition} {produce} for consumption safety?",
            "What quality indicators should I evaluate to determine edibility of this {condition} {produce}?",
            "Based on visual assessment, is this {produce} exhibiting signs of quality deterioration consistent with {condition}?",
            "What organoleptic evaluation would determine the safety of this {condition} {produce}?"
        ]
    },
    
    "fruit_ripeness": {
        "low": [
            "Is this {produce} ripe?", 
            "Is my {produce} ready to eat?",
            "How do I know if this {produce} is ripe?",
            "My {produce} is {condition}, is it ripe?",
            "When will my {produce} be ready to eat?"
        ],
        "medium": [
            "What are the ripeness indicators for this {produce}?",
            "How can I determine the optimal ripeness stage for this {produce}?",
            "Is this {condition} {produce} at peak ripeness?",
            "What sensory characteristics indicate proper ripening in {produce}?"
        ],
        "high": [
            "What physiological changes occur during the ripening process of {produce}?",
            "How do ethylene levels correlate with the ripening stages of this {produce}?",
            "Can you identify the optimal harvest maturity indicators for this {produce} variety?",
            "What biochemical markers would indicate peak ripeness in this {produce}?"
        ]
    },
    
    "plant_disease_identification": {
        "low": [
            "What's wrong with my {produce}?",
            "Does my {produce} have a disease?",
            "Why does my {produce} look {condition}?",
            "Is this {condition} on my {produce} a disease?",
            "My {produce} has spots, what is it?"
        ],
        "medium": [
            "Can you identify the disease affecting this {produce}?",
            "What pathogen might be causing these {condition} symptoms on my {produce}?",
            "Is this {condition} appearance on my {produce} indicative of a specific disease?",
            "What disease causes {produce} leaves to become {condition}?"
        ],
        "high": [
            "Based on these symptomatic presentations, what pathogen is likely infecting this {produce}?",
            "Can you differentiate between potential pathogens causing this {condition} manifestation in {produce}?",
            "What diagnostic protocol would you recommend to confirm this suspected {condition} in {produce}?",
            "Is this {condition} in {produce} consistent with viral, bacterial, or fungal etiology?"
        ]
    },
    
    "pest_management": {
        "low": [
            "What are these bugs on my {produce}?",
            "How do I get rid of bugs on my {produce}?",
            "Are these insects harming my {produce}?",
            "Something's eating my {produce}, what is it?",
            "My {produce} has holes in the leaves, what's causing it?"
        ],
        "medium": [
            "Can you identify the pest affecting my {produce}?",
            "What integrated pest management approach would work for these insects on my {produce}?",
            "How can I control the {condition} on my {produce} plants?",
            "What treatment options exist for managing this pest on my {produce}?"
        ],
        "high": [
            "What biological control agents would be effective against this pest complex in {produce}?",
            "Can you recommend a targeted intervention strategy for this arthropod infestation in {produce}?",
            "What ecological factors are contributing to this pest pressure on my {produce} crop?",
            "How would you assess the economic threshold for treatment of this pest in commercial {produce} production?"
        ]
    },
    
    "storage_recommendations": {
        "low": [
            "How should I store this {produce}?",
            "Can I put {produce} in the fridge?",
            "How long will this {produce} last?",
            "Will my {produce} go bad if I leave it out?",
            "What's the best way to keep {produce} fresh?"
        ],
        "medium": [
            "What are the optimal storage conditions for maintaining quality of {produce}?",
            "How do temperature and humidity affect storage life of {produce}?",
            "What post-harvest handling practices extend shelf life of {produce}?",
            "How can I prevent quality loss during storage of this {produce}?"
        ],
        "high": [
            "What controlled atmosphere parameters would maximize storage potential of this {produce} variety?",
            "How do ethylene management strategies impact quality retention in stored {produce}?",
            "What physiological indicators should be monitored during long-term storage of this {produce}?",
            "Can you recommend optimal modified atmosphere packaging specifications for extending shelf life of {produce}?"
        ]
    },
    
    "cultivation_practices": {
        "low": [
            "How do I grow {produce}?",
            "When should I plant {produce}?",
            "How much water does {produce} need?",
            "Why isn't my {produce} growing well?",
            "What kind of soil is best for {produce}?"
        ],
        "medium": [
            "What cultivation techniques optimize yield for {produce}?",
            "How should I adjust fertilization regimes for {produce} in {condition} soil?",
            "What trellising system works best for {produce}?",
            "How do I implement crop rotation with {produce}?"
        ],
        "high": [
            "What physiological responses occur in {produce} under various irrigation regimes?",
            "How do micronutrient balances affect development of {produce} under {condition} conditions?",
            "What are the optimal photoperiod requirements for flowering induction in {produce}?",
            "Can you explain the relationship between root zone temperature and nutrient uptake efficiency in {produce}?"
        ]
    },
    
    "harvest_timing": {
        "low": [
            "When should I pick my {produce}?",
            "Is my {produce} ready to harvest?",
            "How do I know when to pick {produce}?",
            "What's the best time to harvest {produce}?",
            "My {produce} is {condition}, should I harvest it now?"
        ],
        "medium": [
            "What indicators determine optimal harvest timing for {produce}?",
            "How do I assess maturity for harvesting {produce}?",
            "What changes signal harvest readiness in {produce}?",
            "How does harvest timing affect post-harvest quality of {produce}?"
        ],
        "high": [
            "What physiological maturity parameters should be measured to determine optimal harvest timing for {produce}?",
            "How do environmental factors affect the harvest window for {produce}?",
            "What biochemical changes correspond with optimal harvest maturity in {produce}?",
            "Can you explain the relationship between dry matter content and harvest timing in {produce}?"
        ]
    },
    
    "nutritional_information": {
        "low": [
            "Is {produce} healthy?",
            "What vitamins are in {produce}?",
            "Is {produce} good for you?",
            "How many calories in {produce}?",
            "Does {produce} have a lot of sugar?"
        ],
        "medium": [
            "What nutritional benefits does {produce} provide?",
            "How does the nutrient profile of {produce} compare to other fruits?",
            "What is the micronutrient composition of {produce}?",
            "How does ripeness affect the nutritional value of {produce}?"
        ],
        "high": [
            "What bioactive compounds in {produce} contribute to its antioxidant capacity?",
            "How do cultivation practices affect phytonutrient concentrations in {produce}?",
            "What is the bioavailability of carotenoids in {condition} {produce}?",
            "Can you quantify the changes in vitamin content during ripening stages of {produce}?"
        ]
    },
    
    "variety_identification": {
        "low": [
            "What kind of {produce} is this?",
            "What variety of {produce} do I have?",
            "Is this a special type of {produce}?",
            "What's this {produce} called?",
            "Is this a common {produce} variety?"
        ],
        "medium": [
            "Can you identify this cultivar of {produce}?",
            "What characteristics distinguish this variety of {produce}?",
            "Is this an heirloom or commercial variety of {produce}?",
            "How can I determine the specific variety of this {produce}?"
        ],
        "high": [
            "What morphological traits would help identify this {produce} cultivar?",
            "Can you differentiate between possible varieties based on these phenotypic expressions in this {produce}?",
            "What genetic markers are typically used to classify varieties of {produce}?",
            "How would you distinguish between similar cultivars in this {produce} species?"
        ]
    },
    
    "market_quality": {
        "low": [
            "Is this {produce} good enough to sell?",
            "Will customers buy this {condition} {produce}?",
            "Is this {produce} marketable?",
            "Will stores accept this {produce}?",
            "Is this {produce} good quality for market?"
        ],
        "medium": [
            "Does this {produce} meet commercial quality standards?",
            "What grade would this {condition} {produce} receive?",
            "How would this {produce} be classified in market grading?",
            "Would this {produce} qualify for premium market segments?"
        ],
        "high": [
            "What quantitative and qualitative parameters would determine the market classification of this {produce}?",
            "How do these quality attributes affect the market positioning of this {produce} variety?",
            "What sensory evaluation metrics would influence consumer acceptance of this {condition} {produce}?",
            "By what standards would export markets evaluate this {produce} quality?"
        ]
    },
    
    "out_of_scope": [
        "What's the weather forecast for tomorrow?",
        "How do I reset my phone?",
        "Where is the nearest gas station?",
        "Can you recommend a good restaurant?",
        "What time does the bank close?",
        "How do I file my taxes?",
        "What's the score of the game?",
        "Can you play some music?",
        "How do I change a flat tire?",
        "What's the capital of France?",
        "How do I make bread?",
        "What's the best smartphone to buy?",
        "Can you order me a pizza?",
        "How do I get to the airport?",
        "Who won the Oscar for Best Picture?",
        "What exercises are good for losing weight?",
        "Can you translate this sentence to Spanish?",
        "How much does a house cost in Boston?",
        "What's the meaning of life?",
        "How do I set up a wireless printer?",
        "What stocks should I invest in?",
        "Can you tell me a joke?",
        "How do I train my dog?",
        "What's happening in the news today?",
        "How do I fix my internet connection?",
        "What's a good birthday gift for my mom?",
        "How do I renew my driver's license?",
        "What movies are playing this weekend?",
        "How many calories are in a cheeseburger?",
        "What's the best way to learn French?",
        "How do I start investing in stocks?",
        "What's the difference between a crocodile and an alligator?",
        "Can you help me set up my email?",
        "Where can I watch the latest Marvel movie?",
        "What's the population of Tokyo?",
        "How do I tie a tie?",
        "What's the best exercise for losing belly fat?",
        "Can you translate 'hello' to Japanese?",
        "What are the symptoms of the flu?",
        "How old is the universe?",
        "What's the forecast for next week?",
        "How much does a Tesla cost?",
        "What's the exchange rate between dollars and euros?",
        "Is it going to rain tomorrow?",
        "How do I get a passport?",
        "What are the symptoms of COVID-19?",
        "Can you recommend a good restaurant in New York?",
        "What time is the Super Bowl?",
        "How do I cancel my subscription?",
        "What's the fastest car in the world?",
        "How many calories are in a banana?",
        "Who won the last World Cup?",
        "What's the recipe for chocolate cake?",
        "How do I install Windows on my computer?",
        "When was the moon landing?",
        "Can you recommend a good book to read?",
        "What's the tallest building in the world?",
        "How do I change my profile picture?",
        "What's the best way to remove coffee stains?",
        "Who is the prime minister of Portugal?"
    ]
}

# Function to generate questions based on persona characteristics
def generate_persona_question(persona_key, intent, produce_type=None, condition=None):
    """
    Generate a question based on a specific persona and intent
    
    Parameters:
    persona_key (str): The key of the persona from the PERSONAS dict
    intent (str): The type of question to generate
    produce_type (str, optional): Specific fruit or vegetable
    condition (str, optional): Specific condition
    
    Returns:
    str: A generated question
    dict: Metadata about the question
    """
    persona = PERSONAS[persona_key]
    
    # Select a produce item if not provided
    if not produce_type:
        if intent in ["fruit_quality", "fruit_ripeness", "fruit_storage", "fruit_identification"]:
            produce_type = random.choice(FRUITS)
        else:
            # For plant health, disease questions, etc.
            produce_type = random.choice(FRUITS + VEGETABLES)
    
    # Select a condition if not provided
    if not condition:
        if "disease" in intent or "pest" in intent:
            condition = random.choice(PLANT_DISEASES)
        else:
            condition = random.choice(CONDITIONS)
    
    # Base template selection based on intent and complexity
    templates = INTENT_TEMPLATES[intent]
    
    # Filter by complexity if available
    complexity = persona["complexity"]
    if isinstance(templates, dict) and complexity in templates:
        templates = templates[complexity]
    elif isinstance(templates, dict):
        # If exact complexity not found, use closest available
        if complexity == "very high" and "high" in templates:
            templates = templates["high"]
        elif complexity == "very low" and "low" in templates:
            templates = templates["low"]
        else:
            # Default to medium if specific complexity not available
            templates = templates.get("medium", list(templates.values())[0])
    
    # Select a random template
    template = random.choice(templates)
    
    # Replace placeholders
    question = template.replace("{produce}", produce_type)
    question = question.replace("{condition}", condition)
    
    # Add persona-specific vocabulary
    question = add_persona_vocabulary(question, persona)
    
    # Apply question style
    question = apply_question_style(question, persona["question_style"])
    
    # Add typical speech patterns, including occasional grammatical errors
    question = add_speech_patterns(question, persona_key)
    
    # Metadata for tracking
    metadata = {
        "persona": persona_key,
        "intent": intent,
        "produce": produce_type,
        "condition": condition,
        "template": template,
        "complexity": complexity
    }
    
    return question, metadata

def add_persona_vocabulary(question, persona):
    """Replace generic terms with persona-specific vocabulary"""
    
    # Words that might be replaced
    generic_terms = {
        "problem": ["issue", "condition", "disorder", "disease", "defect", "damage", "abnormality"],
        "good": ["acceptable", "adequate", "satisfactory", "suitable", "appropriate", "desirable", "quality"],
        "bad": ["problematic", "unsuitable", "inadequate", "substandard", "defective", "poor quality"],
        "look": ["appear", "present", "seem", "manifest", "show signs of", "exhibit", "display"],
    }
    
    # 30% chance to replace a word with persona vocabulary
    if random.random() < 0.3 and persona.get("vocabulary"):
        # Select a word from the question that could be replaced
        words = word_tokenize(question)
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Check if word is in our generic terms dictionary
            for generic, alternatives in generic_terms.items():
                if word_lower == generic and random.random() < 0.4:
                    # Replace with persona-specific term if possible
                    persona_vocab = persona["vocabulary"]
                    relevant_terms = [term for term in persona_vocab 
                                     if any(alt in term for alt in alternatives + [generic])]
                    
                    if relevant_terms:
                        replacement = random.choice(relevant_terms)
                        words[i] = replacement
                        break
        
        # 20% chance to insert a domain-specific term somewhere
        if random.random() < 0.2 and len(words) > 3:
            insert_pos = random.randint(1, len(words) - 2)  # Avoid first and last position
            domain_term = random.choice(persona["vocabulary"])
            
            # Add appropriate connector words
            connector = random.choice(["for", "regarding", "concerning", "related to", "about"])
            words.insert(insert_pos, connector)
            words.insert(insert_pos + 1, domain_term)
        
        question = ' '.join(words)
    
    return question

def apply_question_style(question, style_list):
    """Apply the persona's question style"""
    
    # Get base question without punctuation
    base_question = question.rstrip('?!.,')
    
    # Apply style based on style list
    if "direct" in style_list:
        # More imperative, shorter sentences
        if random.random() < 0.3 and len(base_question.split()) > 5:
            # Shorten question
            words = base_question.split()
            question = ' '.join(words[:len(words)//2]) + "?"
    
    if "technical" in style_list:
        # Add technical framing
        if random.random() < 0.4:
            technical_prefix = random.choice([
                "From a technical standpoint, ",
                "Considering the physiological aspects, ",
                "In terms of agricultural science, ",
                "Based on morphological characteristics, "
            ])
            question = technical_prefix + base_question.lower() + "?"
    
    if "analytical" in style_list:
        # Frame as analysis request
        if random.random() < 0.35:
            analytical_frame = random.choice([
                "Could you analyze ",
                "I need an analysis of ",
                "What analysis would explain ",
                "Can you evaluate "
            ])
            question = analytical_frame + base_question.lower() + "?"
    
    # Ensure question ends with proper punctuation
    if not question.endswith(("?", "!", ".")):
        question += "?"
    
    # Capitalize first letter
    question = question[0].upper() + question[1:]
    
    return question

def add_speech_patterns(question, persona_key):
    """Add realistic speech patterns including occasional errors"""
    
    # Different personas have different error rates and patterns
    error_rate = {
        "farmer": 0.15,
        "final_consumer": 0.2,
        "scientist": 0.05,
        "biologist": 0.05,
        "agricultural_engineer": 0.08,
        "inspector": 0.07,
        "quality_controller": 0.1,
        "agricultural_trader": 0.12,
        "environmental_engineer": 0.08,
        "naturalist": 0.1
    }
    
    # Add filler words based on persona
    filler_words = {
        "farmer": ["basically", "like", "ya know", "pretty much", "kinda"],
        "final_consumer": ["like", "um", "sort of", "kind of", "basically", "just"],
        "scientist": ["essentially", "fundamentally", "theoretically", "arguably", "ostensibly"],
        "agricultural_trader": ["basically", "honestly", "frankly", "bottom line", "essentially"]
    }
    
    # Regional dialects and expressions
    regional_terms = {
        "farmer": {
            "vegetable": ["veg", "veggie", "produce", "crop", "greens"],
            "problem": ["trouble", "issue", "bother", "hassle", "predicament"],
            "look": ["appear", "seem", "show up", "present", "come across"]
        },
        "final_consumer": {
            "vegetable": ["veggie", "veg", "greens", "produce", "food"],
            "bad": ["gross", "nasty", "yucky", "dodgy", "sketchy", "off"],
            "good": ["nice", "yummy", "tasty", "fresh", "decent", "fine"]
        }
    }
    
    # Apply filler words (with moderation)
    if persona_key in filler_words and random.random() < 0.15:
        words = question.split()
        insert_pos = random.randint(1, max(2, len(words) - 1))
        filler = random.choice(filler_words[persona_key])
        words.insert(insert_pos, filler)
        question = ' '.join(words)
    
    # Apply grammatical errors based on error rate
    if random.random() < error_rate.get(persona_key, 0.1):
        error_type = random.choice(["typo", "grammar", "punctuation", "capitalization"])
        
        if error_type == "typo" and len(question) > 5:
            # Simple typo: character swap, omission, or duplication
            pos = random.randint(1, len(question) - 2)
            error_subtype = random.choice(["swap", "omit", "duplicate"])
            
            if error_subtype == "swap" and pos < len(question) - 1:
                # Swap two adjacent characters
                chars = list(question)
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                question = ''.join(chars)
            
            elif error_subtype == "omit":
                # Omit a character
                question = question[:pos] + question[pos + 1:]
            
            elif error_subtype == "duplicate":
                # Duplicate a character
                question = question[:pos] + question[pos] + question[pos:]
        
        elif error_type == "grammar":
            # Grammar errors: subject-verb agreement, tense issues
            words = question.split()
            for i, word in enumerate(words):
                if word.lower() in ["is", "are", "was", "were"] and random.random() < 0.4:
                    if word.lower() == "is":
                        words[i] = "are"
                    elif word.lower() == "are":
                        words[i] = "is"
                    elif word.lower() == "was":
                        words[i] = "were"
                    elif word.lower() == "were":
                        words[i] = "was"
                    break
            question = ' '.join(words)
        
        elif error_type == "punctuation" and question.endswith("?"):
            # Missing question mark
            question = question[:-1]
        
        elif error_type == "capitalization" and len(question) > 0:
            # First letter not capitalized
            question = question[0].lower() + question[1:]
    
    # Regional term substitution
    if persona_key in regional_terms and random.random() < 0.2:
        words = word_tokenize(question)
        for i, word in enumerate(words):
            word_lower = word.lower()
            for standard_term, regional_options in regional_terms[persona_key].items():
                if word_lower == standard_term and random.random() < 0.4:
                    words[i] = random.choice(regional_options)
                    break
        question = ' '.join(words)
    
    return question

def generate_large_dataset(size=100000, seed=42):
    """
    Generate a large, diverse dataset of questions with persona and intent labels
    
    Parameters:
    size (int): Target dataset size
    seed (int): Random seed for reproducibility
    
    Returns:
    pandas.DataFrame: DataFrame with questions and metadata
    """
    random.seed(seed)
    
    # Calculate distribution percentages
    persona_distribution = {
        "farmer": 0.15,
        "agricultural_engineer": 0.10,
        "agricultural_trader": 0.08,
        "environmental_engineer": 0.07,
        "inspector": 0.08,
        "quality_controller": 0.09,
        "final_consumer": 0.20,
        "scientist": 0.08,
        "biologist": 0.07,
        "naturalist": 0.08
    }
    
    # Intent distribution (not including out of scope)
    intent_distribution = {
        "fruit_quality": 0.14,
        "fruit_ripeness": 0.12,
        "plant_disease_identification": 0.11,
        "pest_management": 0.10,
        "storage_recommendations": 0.09,
        "cultivation_practices": 0.08,
        "harvest_timing": 0.08,
        "nutritional_information": 0.10,
        "variety_identification": 0.09,
        "market_quality": 0.09
    }
    
    # Percentage of out of scope questions
    out_of_scope_percentage = 0.10
    
    # Calculate sample counts
    in_scope_count = int(size * (1 - out_of_scope_percentage))
    out_of_scope_count = size - in_scope_count
    
    # Prepare data storage
    data = []
    
    # Generate in-scope questions
    for _ in range(in_scope_count):
        # Select persona based on distribution
        persona = random.choices(
            list(persona_distribution.keys()),
            weights=list(persona_distribution.values())
        )[0]
        
        # Select intent based on distribution
        intent = random.choices(
            list(intent_distribution.keys()),
            weights=list(intent_distribution.values())
        )[0]
        
        # Generate question
        question, metadata = generate_persona_question(persona, intent)
        
        # Add to dataset
        data.append({
            "question": question,
            "intent": intent,
            "persona": persona,
            "produce": metadata.get("produce", ""),
            "condition": metadata.get("condition", "")
        })
    
    # Generate out-of-scope questions
    for _ in range(out_of_scope_count):
        # Select a random persona
        persona = random.choice(list(persona_distribution.keys()))
        
        # Select base out-of-scope question
        base_question = random.choice(INTENT_TEMPLATES["out_of_scope"])
        
        # Apply persona speech patterns
        question = add_speech_patterns(base_question, persona)
        
        # Add to dataset
        data.append({
            "question": question,
            "intent": "out_of_scope",
            "persona": persona,
            "produce": "",
            "condition": ""
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["question"])
    
    # If after removing duplicates we're below target size, generate more
    if len(df) < size:
        additional_needed = size - len(df)
        print(f"Generating {additional_needed} additional samples to replace duplicates...")
        additional_df = generate_large_dataset(size=additional_needed, seed=seed+1)
        df = pd.concat([df, additional_df], ignore_index=True)
        df = df.drop_duplicates(subset=["question"])
        df = df.sample(n=min(size, len(df)), random_state=seed).reset_index(drop=True)
    
    return df
def augment_dataset(df, augmentation_factor=0.2):
    """
    Augment the dataset with additional variations
    
    Parameters:
    df (pandas.DataFrame): Original dataset
    augmentation_factor (float): Percentage of data to augment
    
    Returns:
    pandas.DataFrame: Augmented dataset
    """
    print(f"Augmenting {int(len(df) * augmentation_factor)} samples...")
    
    # Select samples to augment
    samples_to_augment = df.sample(frac=augmentation_factor)
    augmented_data = []
    
    # Define augmentation techniques
    augmentation_techniques = [
        "misspelling",
        "word_replacement",
        "punctuation_variation",
        "word_order",
        "contraction"
    ]
    
    for _, row in samples_to_augment.iterrows():
        # Select an augmentation technique
        technique = random.choice(augmentation_techniques)
        
        if technique == "misspelling":
            augmented_question = introduce_misspelling(row['question'])
        elif technique == "word_replacement":
            augmented_question = replace_words_with_synonyms(row['question'])
        elif technique == "punctuation_variation":
            augmented_question = vary_punctuation(row['question'])
        elif technique == "word_order":
            augmented_question = vary_word_order(row['question'])
        elif technique == "contraction":
            augmented_question = apply_contractions(row['question'])
        
        # Add to augmented data
        augmented_data.append({
            "question": augmented_question,
            "intent": row['intent'],
            "persona": row['persona'],
            "produce": row['produce'],
            "condition": row['condition']
        })
    
    # Create DataFrame from augmented data
    augmented_df = pd.DataFrame(augmented_data)
    
    # Combine with original data
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    
    # Remove any duplicates that might have been created
    combined_df = combined_df.drop_duplicates(subset=["question"])
    
    return combined_df
def introduce_misspelling(text):
    """Introduce realistic misspellings to text"""
    words = text.split()
    if len(words) < 2:
        return text
    
    # Common misspelling patterns
    misspelling_patterns = {
        'ie': 'ei',
        'ei': 'ie',
        'a': 'e',
        'e': 'a',
        'to': 'too',
        'too': 'to',
        'your': 'youre',
        'youre': 'your',
        'their': 'there',
        'there': 'their',
        'its': "it's",
        "it's": 'its',
        'affect': 'effect',
        'effect': 'affect',
        'than': 'then',
        'then': 'than',
        'lose': 'loose',
        'loose': 'lose',
        'ible': 'able',
        'able': 'ible',
        'tion': 'sion',
        'sion': 'tion'
    }
    
    # Select 1-2 words to misspell
    num_misspellings = min(random.randint(1, 2), len(words))
    indices_to_misspell = random.sample(range(len(words)), num_misspellings)
    
    for idx in indices_to_misspell:
        word = words[idx].lower()
        
        # Apply common misspelling pattern if applicable
        for pattern, replacement in misspelling_patterns.items():
            if pattern in word and random.random() < 0.7:
                words[idx] = words[idx].replace(pattern, replacement)
                break
        else:
            # If no pattern matched, try letter swap, duplication, or omission
            if len(word) > 3:
                error_type = random.choice(['swap', 'duplicate', 'omit'])
                char_idx = random.randint(1, len(word) - 2)
                
                if error_type == 'swap' and char_idx < len(word) - 1:
                    new_word = list(word)
                    new_word[char_idx], new_word[char_idx + 1] = new_word[char_idx + 1], new_word[char_idx]
                    words[idx] = ''.join(new_word)
                elif error_type == 'duplicate':
                    words[idx] = word[:char_idx] + word[char_idx] + word[char_idx:]
                elif error_type == 'omit':
                    words[idx] = word[:char_idx] + word[char_idx + 1:]
    
    return ' '.join(words)

def replace_words_with_synonyms(text):
    """Replace some words with synonyms"""
    # Load spaCy
    doc = nlp(text)
    
    # Only replace content words with synonyms
    replaceable_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    
    # New text with replacements
    new_tokens = []
    
    for token in doc:
        # Only try to replace content words with more than 3 chars about 30% of the time
        if token.pos_ in replaceable_pos and len(token.text) > 3 and random.random() < 0.3:
            # Try to find a synonym
            synsets = wordnet.synsets(token.text.lower())
            if synsets:
                # Get all lemmas from all synsets
                lemmas = []
                for synset in synsets:
                    lemmas.extend(synset.lemmas())
                
                # Extract unique synonym words
                synonyms = set()
                for lemma in lemmas:
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != token.text.lower():
                        synonyms.add(synonym)
                
                # If we have synonyms, select one randomly
                if synonyms:
                    replacement = random.choice(list(synonyms))
                    
                    # Preserve capitalization
                    if token.text[0].isupper():
                        replacement = replacement.capitalize()
                    
                    new_tokens.append(replacement)
                    continue
        
        # If we didn't replace, keep original token
        new_tokens.append(token.text)
    
    return ' '.join(new_tokens)

def vary_punctuation(text):
    """Vary punctuation and add or remove it"""
    # Determine if this is a question
    is_question = text.rstrip().endswith('?')
    
    # Define possible replacements
    if is_question:
        alternatives = ['?', '??', ' ???', '? ', '? Hmm?', '?!', ' ?']
        ending = random.choice(alternatives)
        
        # Replace the question mark
        if text.endswith('?'):
            text = text[:-1] + ending
        else:
            text = text + ending
    else:
        # Add different punctuation to non-questions
        alternatives = ['.', '...', '!', ' !', ' ...', '']
        
        # Remove any ending punctuation
        text = text.rstrip('.!?')
        
        # Add new punctuation
        text = text + random.choice(alternatives)
    
    # Randomly add a comma somewhere in longer text
    if len(text.split()) > 5 and ',' not in text and random.random() < 0.4:
        words = text.split()
        insert_pos = random.randint(2, len(words) - 2)
        words[insert_pos] += ','
        text = ' '.join(words)
    
    return text

def vary_word_order(text):
    """Slightly vary word order in a way that preserves meaning"""
    doc = nlp(text)
    
    # For questions, we can sometimes move prepositional phrases
    if text.endswith('?'):
        # Find prepositional phrases (simplified approach)
        prep_phrases = []
        current_prep_phrase = []
        
        for token in doc:
            if token.dep_ == 'prep':
                # Start of a prepositional phrase
                current_prep_phrase = [token.i]
            elif current_prep_phrase and token.head.i in current_prep_phrase:
                # Part of the current prepositional phrase
                current_prep_phrase.append(token.i)
            elif current_prep_phrase:
                # End of a prepositional phrase
                if len(current_prep_phrase) > 1:
                    prep_phrases.append((min(current_prep_phrase), max(current_prep_phrase)))
                current_prep_phrase = []
        
        # If we found any prep phrases, try to move one
        if prep_phrases and random.random() < 0.6:
            # Choose a phrase to move
            phrase_to_move = random.choice(prep_phrases)
            
            # Convert to a list of tokens to manipulate
            tokens = [token.text for token in doc]
            
            # Extract the phrase
            phrase_tokens = tokens[phrase_to_move[0]:phrase_to_move[1]+1]
            
            # Remove the phrase from its original position
            tokens = tokens[:phrase_to_move[0]] + tokens[phrase_to_move[1]+1:]
            
            # Decide where to move it (beginning or end)
            if random.random() < 0.5 and not any(t.is_punct for t in doc[:3]):
                # Move to beginning
                tokens = phrase_tokens + [','] + tokens
            else:
                # Move before the question mark
                if tokens[-1] == '?':
                    tokens = tokens[:-1] + phrase_tokens + ['?']
                else:
                    tokens = tokens + phrase_tokens
            
            return ' '.join(tokens)
    
    # For non-questions or if we couldn't move a prep phrase
    # Try swapping adjacent adjectives if they exist
    adjective_positions = [i for i, token in enumerate(doc) if token.pos_ == 'ADJ' and i > 0 and i < len(doc) - 1]
    
    if adjective_positions and random.random() < 0.4:
        adj_pos = random.choice(adjective_positions)
        
        # If the adjective is next to another adjective, swap them
        if doc[adj_pos-1].pos_ == 'ADJ':
            tokens = [token.text for token in doc]
            tokens[adj_pos-1], tokens[adj_pos] = tokens[adj_pos], tokens[adj_pos-1]
            return ' '.join(tokens)
    
    # No changes made
    return text

def apply_contractions(text):
    """Apply or remove contractions"""
    # Common contraction mappings
    contractions = {
        "do not": "don't",
        "does not": "doesn't",
        "did not": "didn't",
        "is not": "isn't",
        "are not": "aren't",
        "was not": "wasn't",
        "were not": "weren't",
        "have not": "haven't",
        "has not": "hasn't",
        "had not": "hadn't",
        "will not": "won't",
        "would not": "wouldn't",
        "cannot": "can't",
        "can not": "can't",
        "could not": "couldn't",
        "should not": "shouldn't",
        "must not": "mustn't",
        "it is": "it's",
        "that is": "that's",
        "they are": "they're",
        "we are": "we're",
        "you are": "you're",
        "he is": "he's",
        "she is": "she's",
        "what is": "what's",
        "who is": "who's",
        "where is": "where's",
        "when is": "when's",
        "how is": "how's",
        "there is": "there's",
        "i am": "I'm",
        "i will": "I'll",
        "you will": "you'll",
        "he will": "he'll",
        "she will": "she'll",
        "we will": "we'll",
        "they will": "they'll",
        "i would": "I'd",
        "you would": "you'd",
        "he would": "he'd",
        "she would": "she'd",
        "we would": "we'd",
        "they would": "they'd",
    }
    
    # Reverse contractions
    reverse_contractions = {v: k for k, v in contractions.items()}
    
    # Decide whether to apply or remove contractions
    if "'" in text and random.random() < 0.6:  # Remove contractions
        for contraction, expansion in reverse_contractions.items():
            if contraction in text:
                text = text.replace(contraction, expansion)
                break  # Only replace one contraction per text
    else:  # Apply contractions
        for expansion, contraction in contractions.items():
            if expansion in text.lower():
                # Find case-sensitive match in original text
                pos = text.lower().find(expansion)
                if pos >= 0:
                    before = text[:pos]
                    after = text[pos + len(expansion):]
                    text = before + contraction + after
                    break  # Only replace one contraction per text
    
    return text

def main():
    """
    Main function to generate and save the dataset
    """
    import time
    start_time = time.time()
    
    print("Agricultural Chatbot Dataset Generator")
    print("=====================================")
    print("Generating a large dataset with diverse agricultural questions across different personas...")
    
    # Set parameters
    dataset_size = 100000
    output_file = "agricultural_chatbot_dataset.csv"
    
    # Generate initial dataset (slightly smaller to leave room for augmentation)
    initial_size = int(dataset_size * 0.95)
    print(f"Generating initial {initial_size} samples...")
    df = generate_large_dataset(size=initial_size)
    
    # Augment the dataset
    df = augment_dataset(df, augmentation_factor=0.2)
    
    # If we have too many samples after augmentation, sample down to target size
    max_attempts = 5
    attempt = 0
    while len(df) < dataset_size and attempt < max_attempts:
        attempt += 1
        print(f"Dataset has {len(df)} samples, target is {dataset_size}. Attempt {attempt} to generate more...")
        additional_needed = dataset_size - len(df)
        additional_df = generate_large_dataset(size=additional_needed, seed=42+attempt)
        df = pd.concat([df, additional_df], ignore_index=True)
        df = df.drop_duplicates(subset=["question"])

    # Then trim if we generated too many
    if len(df) > dataset_size:
        df = df.sample(n=dataset_size, random_state=42).reset_index(drop=True)

    
    # Data quality checks
    print("\nPerforming data quality checks...")
    
    # Check for duplicates
    duplicate_count = df.duplicated(subset=["question"]).sum()
    print(f"Found {duplicate_count} duplicate questions ({duplicate_count/len(df)*100:.2f}%)")
    
    if duplicate_count > 0:
        print("Removing duplicates...")
        df = df.drop_duplicates(subset=["question"])
    
    # Check intent distribution
    print("\nIntent distribution:")
    intent_counts = df['intent'].value_counts()
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count} ({count/len(df)*100:.2f}%)")
    
    # Check persona distribution
    print("\nPersona distribution:")
    persona_counts = df['persona'].value_counts()
    for persona, count in persona_counts.items():
        print(f"  {persona}: {count} ({count/len(df)*100:.2f}%)")
    
    # Check question length distribution
    df['question_length'] = df['question'].apply(len)
    avg_length = df['question_length'].mean()
    min_length = df['question_length'].min()
    max_length = df['question_length'].max()
    print(f"\nQuestion length statistics:")
    print(f"  Average length: {avg_length:.1f} characters")
    print(f"  Minimum length: {min_length} characters")
    print(f"  Maximum length: {max_length} characters")
    
    # Save dataset
    print(f"\nSaving dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Sample questions
    print("\nSample questions from each intent:")
    for intent in df['intent'].unique():
        print(f"\n{intent.upper()}:")
        samples = df[df['intent'] == intent].sample(min(3, df[df['intent'] == intent].shape[0]))
        for _, row in samples.iterrows():
            print(f"  [{row['persona']}] {row['question']}")
    
    # Performance stats
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"\nGeneration complete! Generated {len(df)} unique questions in {generation_time:.1f} seconds.")
    print(f"Average generation rate: {len(df)/generation_time:.1f} questions per second")
    
    # Final diagnostics
    print("\nFinal dataset diagnostics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Total unique questions: {df['question'].nunique()}")
    print(f"  Total unique intents: {df['intent'].nunique()}")
    print(f"  Total unique personas: {df['persona'].nunique()}")
    print(f"  Total unique produce items: {df['produce'].nunique()}")
    print(f"  Dataset generation successful!")


if __name__ == "__main__":
    main()
