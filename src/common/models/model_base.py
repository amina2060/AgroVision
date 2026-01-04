# common/models/model_base.py

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Default input image size
IMG_SIZE = (224, 224)

def build_model(num_classes, img_size=IMG_SIZE, dropout_rate=0.3, learning_rate=0.0001, train_base=False):
    """
    Builds a MobileNetV2-based model for classification.
    
    Parameters:
    - num_classes: int, number of output classes
    - img_size: tuple, input image size (height, width)
    - dropout_rate: float, dropout rate after GAP
    - learning_rate: float, optimizer learning rate
    - train_base: bool, whether to fine-tune base model
    
    Returns:
    - model: compiled Keras Model
    """

    # Load MobileNetV2 base
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*img_size, 3))

    # Freeze base layers if not fine-tuning
    for layer in base_model.layers:
        layer.trainable = train_base

    # Add classification head
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
