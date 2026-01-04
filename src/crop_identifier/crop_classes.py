# src/crop_identifier/crop_classes.py

CLASSES = [
    'apple',
    'cassava',
    'corn',
    'grape',
    'rice',
    'tomato'
]

# Mapping dataset folder names to class names
FOLDER_TO_CLASS = {
    'apple': 'apple',
    'cassava': 'cassava',
    'corn (maize)': 'corn',   # dataset uses 'corn (maize)'
    'grape': 'grape',
    'rice': 'rice',
    'tomato': 'tomato'
}
