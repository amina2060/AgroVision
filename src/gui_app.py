import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import threading
import time
import webbrowser

# -------------------------------
# Import your existing pipeline functions
# -------------------------------
from src.common.crop_identifier import identify_crop
from src.common.analysis.severity import calculate_severity, infection_severity
from src.common.visualization.visualize import show_overlay_with_severity

# -------------------------------
# GUI App - Premium Version
# -------------------------------
class AgroVisionPremiumApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 AgroVision Pro – Multi-Crop Leaf Disease Detection")
        self.root.geometry("1400x800")
        
        # Set window icon (you can add your own icon)
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
            
        self.image_path = None
        self.current_image = None
        self.processing = False
        self.crop_colors = {
            "apple": "#FF6B6B",
            "tomato": "#FF4757",
            "corn": "#FFA502",
            "rice": "#2ED573",
            "grape": "#A55EEA",
            "cassava": "#FF9F43"
        }
        
        # Modern color palette
        self.colors = {
            "bg_dark": "#1A1F2C",
            "bg_light": "#2D3748",
            "primary": "#48D1CC",
            "secondary": "#FF6B8B",
            "accent": "#FFD166",
            "success": "#06D6A0",
            "warning": "#FF9E00",
            "danger": "#EF476F",
            "text_light": "#F7FAFC",
            "text_dark": "#2D3748"
        }
        
        self.root.configure(bg=self.colors["bg_dark"])
        self.setup_styles()
        self.build_ui()
        
    def setup_styles(self):
        """Create custom ttk styles for modern widgets"""
        style = ttk.Style()
        
        # Configure button styles
        style.configure('Primary.TButton', 
                       font=('Segoe UI', 12, 'bold'),
                       padding=10,
                       background=self.colors["primary"],
                       foreground='white')
        style.map('Primary.TButton',
                 background=[('active', self.colors["success"])])
        
        style.configure('Upload.TButton',
                       font=('Segoe UI', 11, 'bold'),
                       padding=8,
                       background=self.colors["secondary"])
        
        # Frame styles
        style.configure('Card.TFrame',
                       background=self.colors["bg_light"],
                       relief='raised',
                       borderwidth=2)
        
    def build_ui(self):
        """Build the extravagant GUI interface"""
        # Header Section
        header_frame = tk.Frame(self.root, bg=self.colors["bg_dark"], height=100)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        # Logo and Title
        logo_frame = tk.Frame(header_frame, bg=self.colors["bg_dark"])
        logo_frame.pack(side="left")
        
        # Decorative leaf emoji
        leaf_label = tk.Label(logo_frame, text="🍃", font=("Segoe UI", 40), 
                             bg=self.colors["bg_dark"], fg=self.colors["primary"])
        leaf_label.pack(side="left", padx=(0, 10))
        
        title_frame = tk.Frame(logo_frame, bg=self.colors["bg_dark"])
        title_frame.pack(side="left")
        
        title = tk.Label(title_frame, text="AGROVISION PRO", 
                        font=("Segoe UI", 32, "bold"), 
                        bg=self.colors["bg_dark"], 
                        fg=self.colors["text_light"])
        title.pack()
        
        subtitle = tk.Label(title_frame, 
                          text="Multi-Crop Leaf Disease Detection & Analysis System",
                          font=("Segoe UI", 14), 
                          bg=self.colors["bg_dark"], 
                          fg=self.colors["primary"])
        subtitle.pack()
        
        # Status indicator
        status_frame = tk.Frame(header_frame, bg=self.colors["bg_dark"])
        status_frame.pack(side="right")
        
        self.status_indicator = tk.Label(status_frame, text="● Ready", 
                                        font=("Segoe UI", 12),
                                        bg=self.colors["bg_dark"],
                                        fg=self.colors["success"])
        self.status_indicator.pack()
        
        # Main Content Area
        main_container = tk.Frame(self.root, bg=self.colors["bg_dark"])
        main_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Left Panel - Image and Controls
        left_panel = tk.Frame(main_container, bg=self.colors["bg_dark"], width=600)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Image Preview Frame with shadow effect
        image_frame = tk.Frame(left_panel, bg=self.colors["bg_light"], 
                              relief="sunken", borderwidth=3)
        image_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        self.image_label = tk.Label(image_frame, text="\n📷 No Image Selected\n\nUpload a leaf image to begin analysis", 
                                   font=("Segoe UI", 16), bg="#3A4355", fg=self.colors["text_light"],
                                   compound="top")
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Controls Frame
        controls_frame = tk.Frame(left_panel, bg=self.colors["bg_dark"])
        controls_frame.pack(fill="x", pady=10)
        
        # Upload Button with icon
        upload_btn = tk.Button(controls_frame, 
                              text="📁 UPLOAD LEAF IMAGE", 
                              font=("Segoe UI", 13, "bold"),
                              bg=self.colors["secondary"],
                              fg="white",
                              relief="raised",
                              borderwidth=0,
                              cursor="hand2",
                              height=2,
                              command=self.upload_image)
        upload_btn.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        # Analyze Button with gradient effect
        analyze_btn = tk.Button(controls_frame,
                               text="🔬 START ANALYSIS", 
                               font=("Segoe UI", 13, "bold"),
                               bg=self.colors["primary"],
                               fg="white",
                               relief="raised",
                               borderwidth=0,
                               cursor="hand2",
                               height=2,
                               command=self.analyze)
        analyze_btn.pack(side="left", fill="x", expand=True)
        
        # Right Panel - Results
        right_panel = tk.Frame(main_container, bg=self.colors["bg_dark"], width=600)
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # Results Header
        results_header = tk.Label(right_panel, text="ANALYSIS RESULTS", 
                                 font=("Segoe UI", 20, "bold"),
                                 bg=self.colors["bg_dark"],
                                 fg=self.colors["accent"])
        results_header.pack(pady=(0, 15))
        
        # Progress Animation Frame
        self.progress_frame = tk.Frame(right_panel, bg=self.colors["bg_light"], height=80)
        self.progress_frame.pack(fill="x", pady=(0, 15))
        self.progress_frame.pack_propagate(False)
        
        self.progress_text = tk.Label(self.progress_frame, text="System Ready",
                                     font=("Segoe UI", 14),
                                     bg=self.colors["bg_light"],
                                     fg=self.colors["text_light"])
        self.progress_text.pack(expand=True)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate')
        
        # Results Cards Container
        results_cards = tk.Frame(right_panel, bg=self.colors["bg_dark"])
        results_cards.pack(fill="both", expand=True)
        
        # Crop Card
        self.crop_card = self.create_card(results_cards, "🌱 CROP IDENTIFICATION", 
                                         "Waiting for analysis...")
        self.crop_card.pack(fill="x", pady=(0, 10))
        
        # Disease Card
        self.disease_card = self.create_card(results_cards, "🦠 DISEASE DETECTION", 
                                           "No disease data available")
        self.disease_card.pack(fill="x", pady=(0, 10))
        
        # Severity Card
        self.severity_card = self.create_card(results_cards, "📊 SEVERITY ANALYSIS", 
                                            "Severity level unknown")
        self.severity_card.pack(fill="x", pady=(0, 10))
        
        # Confidence Meter
        confidence_frame = tk.Frame(results_cards, bg=self.colors["bg_light"])
        confidence_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(confidence_frame, text="🎯 CONFIDENCE METER", 
                font=("Segoe UI", 14, "bold"),
                bg=self.colors["bg_light"],
                fg=self.colors["text_light"]).pack(anchor="w", padx=15, pady=(10, 5))
        
        self.confidence_meter = tk.Canvas(confidence_frame, height=30, 
                                         bg=self.colors["bg_light"],
                                         highlightthickness=0)
        self.confidence_meter.pack(fill="x", padx=15, pady=(0, 10))
        
        # Bottom Panel - Summary and Actions
        bottom_panel = tk.Frame(self.root, bg=self.colors["bg_dark"])
        bottom_panel.pack(fill="x", padx=20, pady=(10, 20))
        
        # Summary Card
        summary_frame = tk.Frame(bottom_panel, bg=self.colors["bg_light"], 
                                relief="raised", borderwidth=2)
        summary_frame.pack(fill="both", expand=True)
        
        summary_header = tk.Label(summary_frame, text="📋 DIAGNOSIS SUMMARY", 
                                 font=("Segoe UI", 16, "bold"),
                                 bg=self.colors["bg_light"],
                                 fg=self.colors["accent"])
        summary_header.pack(pady=(15, 10))
        
        self.summary_text = tk.Text(summary_frame, height=6,
                                   font=("Segoe UI", 12),
                                   bg=self.colors["bg_light"],
                                   fg=self.colors["text_light"],
                                   wrap="word",
                                   relief="flat")
        self.summary_text.pack(fill="both", padx=20, pady=(0, 15))
        self.summary_text.insert("1.0", "Upload an image and click 'Start Analysis' to get diagnosis summary.")
        self.summary_text.config(state="disabled")
        
        # Action Buttons Frame
        action_frame = tk.Frame(bottom_panel, bg=self.colors["bg_dark"])
        action_frame.pack(fill="x", pady=(10, 0))
        
        # Save Report Button
        save_btn = tk.Button(action_frame, text="💾 SAVE REPORT",
                            font=("Segoe UI", 11),
                            bg="#4ECDC4",
                            fg="white",
                            command=self.save_report)
        save_btn.pack(side="left", padx=(0, 10))
        
        # View Overlay Button
        overlay_btn = tk.Button(action_frame, text="🔍 VIEW OVERLAY",
                               font=("Segoe UI", 11),
                               bg="#FF9F43",
                               fg="white",
                               command=self.show_overlay)
        overlay_btn.pack(side="left", padx=10)
        
        # Export Button
        export_btn = tk.Button(action_frame, text="📤 EXPORT DATA",
                              font=("Segoe UI", 11),
                              bg="#6C5CE7",
                              fg="white",
                              command=self.export_data)
        export_btn.pack(side="left", padx=10)
        
        # Help Button
        help_btn = tk.Button(action_frame, text="❓ HELP",
                            font=("Segoe UI", 11),
                            bg="#00CEC9",
                            fg="white",
                            command=self.show_help)
        help_btn.pack(side="right")
        
        # Footer
        footer = tk.Label(self.root, 
                         text="© 2024 AgroVision Pro | Multi-Crop Leaf Disease Detection System",
                         font=("Segoe UI", 10),
                         bg=self.colors["bg_dark"],
                         fg=self.colors["text_light"])
        footer.pack(side="bottom", pady=5)
    
    def create_card(self, parent, title, default_text):
        """Create a stylish result card"""
        card = tk.Frame(parent, bg=self.colors["bg_light"], relief="groove", borderwidth=1)
        
        # Title
        tk.Label(card, text=title, 
                font=("Segoe UI", 12, "bold"),
                bg=self.colors["bg_light"],
                fg=self.colors["primary"]).pack(anchor="w", padx=15, pady=(10, 5))
        
        # Content
        content = tk.Label(card, text=default_text,
                          font=("Segoe UI", 14),
                          bg=self.colors["bg_light"],
                          fg=self.colors["text_light"],
                          wraplength=400,
                          justify="left")
        content.pack(fill="x", padx=15, pady=(0, 10))
        
        # Store reference to update later
        if "CROP" in title:
            self.result_crop_label = content
        elif "DISEASE" in title:
            self.result_disease_label = content
        elif "SEVERITY" in title:
            self.result_severity_label = content
        
        return card
    
    def update_confidence_meter(self, confidence):
        """Update the confidence meter with color coding"""
        self.confidence_meter.delete("all")
        
        width = 400
        height = 20
        x = 10
        y = 5
        
        # Draw background
        self.confidence_meter.create_rectangle(x, y, x + width, y + height, 
                                             fill="#4A5568", outline="")
        
        # Draw confidence bar
        bar_width = int(width * confidence / 100)
        
        # Color based on confidence
        if confidence >= 70:
            color = self.colors["success"]
        elif confidence >= 40:
            color = self.colors["warning"]
        else:
            color = self.colors["danger"]
        
        self.confidence_meter.create_rectangle(x, y, x + bar_width, y + height,
                                             fill=color, outline="")
        
        # Draw text
        confidence_text = f"{confidence:.1f}%"
        if confidence >= 70:
            level = "HIGH CONFIDENCE 🟢"
        elif confidence >= 40:
            level = "MEDIUM CONFIDENCE 🟡"
        else:
            level = "LOW CONFIDENCE 🔴"
        
        self.confidence_meter.create_text(x + width/2, y + height/2,
                                         text=f"{confidence_text} | {level}",
                                         fill="white",
                                         font=("Segoe UI", 10, "bold"))
    
    def update_progress(self, message):
        """Update progress text"""
        self.progress_text.config(text=message)
        self.root.update()
    
    def start_progress_animation(self):
        """Start progress bar animation"""
        self.progress_bar.pack(fill="x", padx=20, pady=(0, 10))
        self.progress_bar.start()
    
    def stop_progress_animation(self):
        """Stop progress bar animation"""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
    
    def upload_image(self):
        """Upload and preview image"""
        self.image_path = filedialog.askopenfilename(
            title="Select Leaf Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if not self.image_path:
            return
        
        # Update status
        self.status_indicator.config(text="● Image Loaded", fg=self.colors["primary"])
        
        # Load and display image
        img = Image.open(self.image_path)
        self.current_image = img
        
        # Resize maintaining aspect ratio
        img.thumbnail((550, 400))
        
        # Create a bordered image
        bordered = Image.new('RGB', (img.width + 20, img.height + 20), '#3A4355')
        bordered.paste(img, (10, 10))
        
        # Add some decoration
        draw = ImageDraw.Draw(bordered)
        draw.rectangle([0, 0, bordered.width-1, bordered.height-1], 
                      outline=self.colors["primary"], width=2)
        
        self.tk_img = ImageTk.PhotoImage(bordered)
        self.image_label.configure(image=self.tk_img, text="")
    
    def analyze(self):
        """Run analysis in separate thread"""
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        
        if self.processing:
            return
        
        self.processing = True
        self.status_indicator.config(text="● Processing...", fg=self.colors["warning"])
        
        # Start progress animation
        self.start_progress_animation()
        
        # Run analysis in separate thread
        thread = threading.Thread(target=self.run_analysis_pipeline)
        thread.daemon = True
        thread.start()
    
    def run_analysis_pipeline(self):
        """Run the complete analysis pipeline"""
        try:
            # Step 0: Initializing
            self.update_progress("🔍 Initializing analysis system...")
            time.sleep(0.5)
            
            # Step 1: Crop Identification
            self.update_progress("🌱 Identifying crop type...")
            time.sleep(0.3)
            crop_name, confidence = identify_crop(self.image_path)
            
            if crop_name == "unknown":
                self.root.after(0, self.analysis_failed, "Crop could not be identified.")
                return
            
            # Update crop card with color coding
            crop_color = self.crop_colors.get(crop_name, self.colors["primary"])
            crop_display = f"Crop: {crop_name.upper()}\nConfidence: {confidence:.1f}%"
            self.root.after(0, lambda: self.result_crop_label.config(
                text=crop_display, fg=crop_color))
            
            # Update confidence meter
            self.root.after(0, lambda: self.update_confidence_meter(confidence * 100))
            
            # Step 2: Disease Prediction
            self.update_progress("🦠 Detecting diseases...")
            time.sleep(0.3)
            
            # Import appropriate module based on crop
            disease_name = self.predict_disease(crop_name)
            if not disease_name:
                self.root.after(0, self.analysis_failed, "Disease prediction failed.")
                return
            
            self.root.after(0, lambda: self.result_disease_label.config(
                text=f"Disease: {disease_name.upper()}",
                fg=self.colors["danger"] if "healthy" not in disease_name.lower() else self.colors["success"]))
            
            # Step 3: Segmentation
            self.update_progress("📐 Analyzing leaf segmentation...")
            time.sleep(0.3)
            
            mask, overlay = self.perform_segmentation(crop_name)
            if mask is None:
                self.root.after(0, self.analysis_failed, "Segmentation failed.")
                return
            
            # Step 4: Severity Calculation
            self.update_progress("📊 Calculating severity...")
            time.sleep(0.3)
            
            percent = calculate_severity(mask)
            level = infection_severity(percent)
            
            # Color code severity
            if level.lower() == "minor":
                severity_color = self.colors["success"]
            elif level.lower() == "moderate":
                severity_color = self.colors["warning"]
            else:
                severity_color = self.colors["danger"]
            
            severity_text = f"Infection: {percent:.2f}%\nSeverity: {level.upper()}"
            self.root.after(0, lambda: self.result_severity_label.config(
                text=severity_text, fg=severity_color))
            
            # Step 5: Update Summary
            self.update_progress("📋 Generating final report...")
            time.sleep(0.3)
            
            self.root.after(0, lambda: self.update_summary(crop_name, disease_name, percent, level))
            
            # Step 6: Finalize
            self.root.after(0, self.analysis_completed, crop_name, disease_name, percent, level)
            
        except Exception as e:
            self.root.after(0, self.analysis_failed, str(e))
    
    def predict_disease(self, crop_name):
        """Predict disease based on crop type"""
        try:
            if crop_name == "apple":
                from src.apple.apple_predict import predict_apple
                from src.apple.apple_classes import CLASSES
                disease_index = predict_apple(self.image_path)
                return CLASSES[disease_index]
                
            elif crop_name == "tomato":
                from src.tomato.tomato_predict import predict_tomato
                from src.tomato.tomato_classes import CLASSES
                disease_index = predict_tomato(self.image_path)
                return CLASSES[disease_index]
                
            elif crop_name == "corn":
                from src.corn.corn_predict import predict_corn
                from src.corn.corn_classes import CLASSES
                disease_index = predict_corn(self.image_path)
                return CLASSES[disease_index]
                
            elif crop_name == "rice":
                from src.rice.rice_predict import predict_rice
                from src.rice.rice_classes import CLASSES
                disease_index = predict_rice(self.image_path)
                return CLASSES[disease_index]
                
            elif crop_name == "grape":
                from src.grape.grape_predict import predict_grape
                from src.grape.grape_classes import CLASSES
                disease_index = predict_grape(self.image_path)
                return CLASSES[disease_index]
                
            elif crop_name == "cassava":
                from src.cassava.cassava_predict import predict_cassava
                from src.cassava.cassava_classes import CLASSES
                disease_index = predict_cassava(self.image_path)
                return CLASSES[disease_index]
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Unknown"
    
    def perform_segmentation(self, crop_name):
        """Perform segmentation based on crop type"""
        try:
            segmentation_map = {
                "apple": ("src.apple.segmentation.apple_segmentation", "segment_apple_leaf"),
                "tomato": ("src.tomato.segmentation.tomato_segmentation", "segment_tomato_leaf"),
                "corn": ("src.corn.segmentation.corn_segmentation", "segment_corn_leaf"),
                "rice": ("src.rice.segmentation.rice_segmentation", "segment_rice_leaf"),
                "grape": ("src.grape.segmentation.grape_segmentation", "segment_grape_leaf"),
                "cassava": ("src.cassava.segmentation.cassava_segmentation", "segment_cassava_leaf")
            }
            
            module_path, func_name = segmentation_map[crop_name]
            seg_module = __import__(module_path, fromlist=[func_name])
            seg_func = getattr(seg_module, func_name)
            return seg_func(self.image_path)
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            return None, None
    
    def update_summary(self, crop, disease, percent, level):
        """Update the summary text box"""
        summary = f"""🌿 CROP: {crop.upper()}
🦠 DETECTED DISEASE: {disease.upper()}
📊 INFECTION LEVEL: {percent:.2f}% ({level.upper()})
⚡ SEVERITY STATUS: {self.get_severity_icon(level)}
💡 RECOMMENDED ACTION: {self.get_recommendation(level, disease)}

Analysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", summary)
        
        # Color code based on severity
        if level.lower() == "minor":
            self.summary_text.config(fg=self.colors["success"])
        elif level.lower() == "moderate":
            self.summary_text.config(fg=self.colors["warning"])
        else:
            self.summary_text.config(fg=self.colors["danger"])
        
        self.summary_text.config(state="disabled")
    
    def get_severity_icon(self, level):
        """Get appropriate icon for severity level"""
        if level.lower() == "minor":
            return "🟩 MINOR - Leaf mostly healthy"
        elif level.lower() == "moderate":
            return "🟨 MODERATE - Treatment recommended"
        else:
            return "🟥 SEVERE - Urgent action required"
    
    def get_recommendation(self, level, disease):
        """Get recommendation based on severity and disease"""
        if "healthy" in disease.lower():
            return "No action needed. Continue regular monitoring."
        
        recommendations = {
            "minor": "Apply organic fungicide. Monitor for changes.",
            "moderate": "Remove infected leaves. Apply chemical treatment.",
            "severe": "Quarantine plant. Apply intensive treatment. Consider replacement."
        }
        
        return recommendations.get(level.lower(), "Consult agricultural expert.")
    
    def analysis_completed(self, crop, disease, percent, level):
        """Handle analysis completion"""
        self.stop_progress_animation()
        self.update_progress("✅ Analysis Completed Successfully!")
        self.status_indicator.config(text="● Analysis Complete", fg=self.colors["success"])
        self.processing = False
        
        # Show success message
        messagebox.showinfo("Analysis Complete", 
                          f"✅ Analysis completed!\n\n"
                          f"Crop: {crop}\n"
                          f"Disease: {disease}\n"
                          f"Severity: {level} ({percent:.2f}%)")
    
    def analysis_failed(self, error_msg):
        """Handle analysis failure"""
        self.stop_progress_animation()
        self.update_progress("❌ Analysis Failed")
        self.status_indicator.config(text="● Error", fg=self.colors["danger"])
        self.processing = False
        
        messagebox.showerror("Analysis Error", 
                           f"Analysis failed:\n{error_msg}")
    
    def show_overlay(self):
        """Show segmentation overlay"""
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload and analyze an image first.")
            return
        
        # This would use your existing show_overlay_with_severity function
        # For now, show a message
        messagebox.showinfo("Overlay", "Overlay visualization would open here.")
    
    def save_report(self):
        """Save analysis report"""
        if not hasattr(self, 'summary_text'):
            messagebox.showwarning("No Data", "Please analyze an image first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write("AGROVISION PRO - ANALYSIS REPORT\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(self.summary_text.get("1.0", "end"))
                messagebox.showinfo("Success", f"Report saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report: {str(e)}")
    
    def export_data(self):
        """Export data in various formats"""
        messagebox.showinfo("Export", "Export feature would open here.")
    
    def show_help(self):
        """Show help information"""
        help_text = """🌿 AGROVISION PRO - HELP GUIDE

1. UPLOAD: Click 'Upload Leaf Image' to select a leaf image
2. ANALYZE: Click 'Start Analysis' to begin disease detection
3. RESULTS: View crop, disease, and severity analysis
4. ACTIONS: Save report, view overlay, or export data

SUPPORTED CROPS:
• Apple, Tomato, Corn, Rice, Grape, Cassava

NOTE: For best results, use clear images with good lighting.
"""
        messagebox.showinfo("Help Guide", help_text)


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    
    # Add some window effects
    root.attributes('-alpha', 0.96)  # Slight transparency
    
    app = AgroVisionPremiumApp(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()