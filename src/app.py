import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class AgriSightApp:
    def __init__(self, model_path, class_mapping_path):
        """Initialize the application with model and class mappings"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            logging.info("Model loaded successfully")
            with open(class_mapping_path, 'r') as f:
                self.class_mapping = json.load(f)
            logging.info("Class mapping loaded successfully")
            self.disease_info = self._load_disease_info()
            logging.info("Disease information loaded successfully")
        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            raise
    
    def _load_disease_info(self):
        """Load disease information including symptoms and treatments"""
        return {
            "Apple___Apple_scab": {
                "symptoms": "- Dark olive-green spots on leaves\n- Dark, scabby lesions on fruits\n- Deformed fruits\n- Premature leaf drop",
                "treatment": "1. Remove infected leaves and fruits\n2. Apply fungicides in early spring\n3. Maintain good air circulation\n4. Plant resistant varieties"
            },
            "Apple___Black_rot": {
                "symptoms": "- Purple spots on leaves\n- Rotting fruit with black centers\n- Cankers on branches\n- Leaf yellowing",
                "treatment": "1. Prune infected branches\n2. Remove mummified fruits\n3. Apply fungicides during growing season\n4. Improve orchard sanitation"
            },
            "Apple___Cedar_apple_rust": {
                "symptoms": "- Bright orange spots on leaves\n- Yellow-orange lesions\n- Deformed fruits\n- Early defoliation",
                "treatment": "1. Remove nearby cedar trees\n2. Apply protective fungicides\n3. Plant resistant varieties\n4. Maintain tree vigor"
            },
            "Corn_(maize)___Common_rust_": {
                "symptoms": "- Small, round, rust-colored spots\n- Spots on both leaf surfaces\n- Chlorotic areas around pustules\n- Reduced photosynthesis",
                "treatment": "1. Plant resistant hybrids\n2. Apply fungicides early\n3. Monitor fields regularly\n4. Maintain proper spacing"
            },
            "Grape___Black_rot": {
                "symptoms": "- Brown circular lesions on leaves\n- Black rotted fruits\n- Tiny black dots in lesions\n- Withered fruits",
                "treatment": "1. Remove infected plant material\n2. Apply fungicides preventively\n3. Improve air circulation\n4. Proper canopy management"
            },
            "Potato___Early_blight": {
                "symptoms": "- Dark brown spots with rings\n- Yellowing around lesions\n- Lower leaves affected first\n- Stem lesions",
                "treatment": "1. Rotate crops\n2. Remove infected leaves\n3. Apply fungicides\n4. Maintain plant nutrition"
            },
            "Potato___Late_blight": {
                "symptoms": "- Water-soaked black spots\n- White fuzzy growth\n- Rapid tissue death\n- Tuber rot",
                "treatment": "1. Use resistant varieties\n2. Apply protective fungicides\n3. Monitor weather conditions\n4. Destroy infected plants"
            },
            "Tomato___Bacterial_spot": {
                "symptoms": "- Small dark spots on leaves\n- Spots with yellow halos\n- Scabby lesions on fruits\n- Defoliation",
                "treatment": "1. Use disease-free seeds\n2. Apply copper-based sprays\n3. Avoid overhead irrigation\n4. Practice crop rotation"
            },
            "Tomato___Early_blight": {
                "symptoms": "- Dark brown spots with rings\n- Yellowing around lesions\n- Lower leaves affected first\n- Stem cankers",
                "treatment": "1. Remove infected leaves\n2. Improve air circulation\n3. Apply fungicides\n4. Mulch around plants"
            },
            "Tomato___Late_blight": {
                "symptoms": "- Dark brown spots on leaves\n- White fungal growth\n- Rapid tissue death\n- Fruit rot",
                "treatment": "1. Remove infected plants\n2. Apply fungicides preventively\n3. Improve drainage\n4. Space plants properly"
            },
            "Tomato___Leaf_Mold": {
                "symptoms": "- Yellow spots on upper leaf surface\n- Olive-green mold underneath\n- Leaf curling\n- Defoliation",
                "treatment": "1. Reduce humidity\n2. Improve ventilation\n3. Apply fungicides\n4. Remove infected leaves"
            },
            "Tomato___Septoria_leaf_spot": {
                "symptoms": "- Small circular spots\n- Dark centers with light borders\n- Lower leaves first\n- Severe defoliation",
                "treatment": "1. Remove infected leaves\n2. Mulch around plants\n3. Apply fungicides\n4. Avoid overhead watering"
            }
        }
    
    def preprocess_image(self, image):
        """Preprocess the input image for model prediction"""
        try:
            # Convert input to RGB numpy array
            if isinstance(image, str):
                image_np = np.array(Image.open(image).convert('RGB'))
            elif isinstance(image, Image.Image):
                image_np = np.array(image.convert('RGB'))
            elif isinstance(image, np.ndarray):
                # Assume RGB if coming from Gradio; only fix grayscale/alpha
                if image.ndim == 2:
                    image_np = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[-1] == 4:
                    image_np = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                else:
                    image_np = image  # already RGB
            else:
                # Fallback handling
                try:
                    image_np = np.array(Image.fromarray(image).convert('RGB'))
                except Exception:
                    image_np = np.array(image)

            viz_image = image_np.copy()

            # Determine target input size from model
            input_shape = getattr(self.model, 'input_shape', None)
            if input_shape and len(input_shape) == 4 and input_shape[1] and input_shape[2]:
                h, w = int(input_shape[1]), int(input_shape[2])
                target_size = (w, h)  # cv2.resize expects (width, height)
            else:
                target_size = (128, 128)  # Fallback to 128x128 (width, height)

            # Resize and normalize
            image_resized = cv2.resize(image_np, target_size)
            image_normalized = (image_resized.astype(np.float32)) / 255.0
            return np.expand_dims(image_normalized, axis=0), viz_image
        except Exception as e:
            logging.error(f"Error during image preprocessing: {str(e)}")
            raise
    
    def create_visualization(self, image, class_name, confidence):
        """Create an annotated visualization of the prediction"""
        try:
            height, width = image.shape[:2]
            
            # Create semi-transparent overlay for text background
            overlay = image.copy()
            cv2.rectangle(overlay, (0, height-60), (width, height), (0, 0, 0), -1)
            
            # Add prediction text
            confidence_pct = float(confidence.strip('%'))
            text = f"{class_name}: {confidence_pct:.1f}%"
            cv2.putText(overlay, text, (10, height-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Blend overlay with original image
            return cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        except Exception as e:
            logging.error(f"Error during visualization creation: {str(e)}")
            raise
    
    def predict(self, image):
        """Make prediction on the input image with detailed analysis"""
        try:
            if image is None:
                return {
                    "visualization": None,
                    "disease": "No image provided",
                    "confidence": "0%",
                    "symptoms": "",
                    "treatment": ""
                }

            # Preprocess image
            processed_image, original_image = self.preprocess_image(image)

            # Get prediction probabilities
            predictions = self.model.predict(processed_image, verbose=0)
            probs = predictions[0].astype(np.float64)
            # Apply softmax if outputs look like logits (sum not ~1)
            if not np.isclose(probs.sum(), 1.0, atol=1e-3):
                probs = tf.nn.softmax(probs).numpy()

            predicted_class_idx = int(np.argmax(probs))
            confidence = float(probs[predicted_class_idx])

            # Debug: log top-3 classes for sanity check
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3_info = ", ".join([
                f"{self.class_mapping.get(str(i), str(i))}: {probs[i]*100:.1f}%" for i in top3_idx
            ])
            logging.info(f"Top-3 predictions: {top3_info}")

            # Get class name and format confidence
            class_name = self.class_mapping.get(str(predicted_class_idx), f"Class_{predicted_class_idx}")
            confidence_str = f"{confidence * 100:.2f}%"

            # Get disease information
            disease_info = self.disease_info.get(class_name, {
                "symptoms": "Information not available",
                "treatment": "Please consult an agricultural expert"
            })

            # Create visualization
            visualization = self.create_visualization(original_image, class_name, confidence_str)

            logging.info(f"Successful prediction: {class_name} with confidence {confidence_str}")

            return {
                "visualization": visualization,
                "disease": class_name,
                "confidence": confidence_str,
                "symptoms": disease_info["symptoms"],
                "treatment": disease_info["treatment"]
            }
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return {
                "visualization": None,
                "disease": "Error during prediction",
                "confidence": "0%",
                "symptoms": f"An error occurred: {str(e)}",
                "treatment": "Please try again or contact support"
            }
    
    def create_interface(self):
        """Create and launch the Gradio interface with queue disabled on the click event."""
        def predict_with_progress(image):
            if image is None:
                return [None, "No image provided", "0%", "", ""]
            try:
                result = self.predict(image)
                return [
                    result["visualization"],
                    result["disease"],
                    result["confidence"],
                    result["symptoms"],
                    result["treatment"]
                ]
            except Exception as e:
                logging.error(f"Prediction error: {str(e)}")
                return [None, "Error during prediction", "0%",
                        f"An error occurred: {str(e)}",
                        "Please try again or contact support"]

        with gr.Blocks(theme=gr.themes.Soft(), analytics_enabled=False) as demo:
            gr.Markdown("# AgriSight: Plant Disease Detection")
            gr.Markdown("Upload an image of a plant leaf to detect diseases and get treatment recommendations.\n\nSupported Plants: Apple, Tomato, Potato, Corn, and more.")
            image_input = gr.Image(label="Upload Leaf Image", type="pil")
            analyze_btn = gr.Button("Analyze")
            viz_output = gr.Image(label="Analysis Visualization")
            disease_output = gr.Textbox(label="Detected Disease")
            confidence_output = gr.Textbox(label="Confidence Score")
            symptoms_output = gr.Textbox(label="Symptoms", lines=4)
            treatment_output = gr.Textbox(label="Recommended Treatment", lines=4)

            analyze_btn.click(
                predict_with_progress,
                inputs=image_input,
                outputs=[viz_output, disease_output, confidence_output, symptoms_output, treatment_output],
                queue=False
            )

        return demo

if __name__ == "__main__":
    try:
        # Get absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "final_model.h5")
        class_mapping_path = os.path.join(base_dir, "models", "class_mapping.json")
        
        # Create and launch app
        app = AgriSightApp(model_path, class_mapping_path)
        interface = app.create_interface()
        interface.launch(
            server_name="127.0.0.1",
            server_port=7862,
            share=False,  # Disable sharing to reduce API calls
            debug=True,
            show_error=True,
            quiet=True  # Reduce logging noise
        )
    except Exception as e:
        logging.error(f"Error during app startup: {str(e)}")
        raise