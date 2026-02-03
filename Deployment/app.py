import streamlit as st 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms.v2 as T
import cv2


st.set_page_config(page_title="Skin Lesion Classifier", layout='wide')  
st.header("Skin Lesion Classifier")    
st.subheader("Upload a skin lesion image to classify it")

st.markdown("""
<style>
div[data-testid="stSidebarHeader"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("About: ")
    st.header("Skin Lesion Classifier, But Mainly Focusing On Melanoma.")
    st.write("This Model Performed: ")
    st.success("The Model Achieved Over 84% Recall On Melanoma Predictions.")
    st.success("75% Accuracy On Average.")
    
    st.markdown("---")
    st.info("""
    This system classifies skin lesions into 7 types:
    - **mel**: Melanoma (dangerous)
    - **bcc**: Basal Cell Carcinoma
    - **nv**: Melanocytic Nevi (benign)
    - **bkl**: Benign Keratosis
    - **akiec**: Actinic Keratosis
    - **vasc**: Vascular Lesions
    - **df**: Dermatofibroma
    """)

    
    st.header("Techinal Details")
    st.markdown('''
            - HAM10000 Dataset
            - Imange Classifer
            - The Model Was Built On ResNet18 Pretrained Model
            - User Grad-CAM For Better Understanding
            ''')
    
    

from PIL import Image

tab1, tab2 = st.tabs(["📤 Upload Your Own Image", "🖼️ Try Sample Images"])
col1, col2 = st.columns(2)
with tab1:
    uploaded_files = st.file_uploader(
        "Upload data", accept_multiple_files=False, type=["jpg", "jpeg", "png"]
    )
    with col1:
        if uploaded_files:
            st.subheader("Original Image")
            df = Image.open(uploaded_files)
            st.image(df)

@st.cache_resource
def loading_model():
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 7)
    weights = torch.load('models/model.pth', map_location='cpu')
    model.load_state_dict(weights)
    model.eval()
    return model
    
with st.spinner("Loading model..."):
    model = loading_model()
st.info("Model loaded successfully!")

img_transform = T.Compose([
    T.Resize((224,224)), 
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label = {
    0: 'Melanocytic Nevi (nv)',
    1: 'Melanoma (mel)',
    2: 'Benign Keratosis (bkl)',
    3: 'Basal Cell Carcinoma (bcc)',
    4: 'Actinic Keratosis (akiec)',
    5: 'Vascular Lesions (vasc)',
    6: 'Dermatofibroma (df)'
}
def prediction(df):
    img = img_transform(df)
    img = img.unsqueeze(dim=0)
    with torch.no_grad():
        logits = model(img)
        y_pred_idx = logits.argmax(dim=1)
        y_pred = logits[0, y_pred_idx]
        
    st.success(f"Predicted Cancer:{label[y_pred_idx.item()]}")
    st.success(f"Confidence Score:{torch.softmax(logits,dim=1)[0,y_pred_idx]}")
    st.markdown("#### Probability Distribution")
    prob_df = pd.DataFrame({
        'Class': [label[i] for i in range(7)],
        'Probability': torch.softmax(logits,dim=1)[0].cpu().numpy()
    }).sort_values('Probability', ascending=False)
    st.dataframe(prob_df, width='content')
    return img
    
with col2:
    if uploaded_files:
       img = prediction(df)


    
def get_activation(name): 
    def hook(model, input, output):  
        activation[name] = output 
    return hook

def get_gradient(name):
    def hook(grad):
        gradient[name] = grad
    return hook

activation = {}
gradient = {}


def grad_cam(model, imgs): 
    model.eval()
    imgs = imgs
    
    model.layer4.register_forward_hook(get_activation('layer4'))
    imgs = imgs.unsqueeze(dim=0).requires_grad_()
    output = model(imgs) 
    activation['layer4'].register_hook(get_gradient('layer4'))
    pred = output.argmax(dim=1)
    score = output[0,pred]
    
    model.zero_grad()
    score.backward()

    activations = activation['layer4'].detach()
    gradients = gradient['layer4'].detach()

    weights = gradients.mean(dim=(2,3))
    weights = weights 

    cam = (weights.unsqueeze(-1).unsqueeze(-1) * activations).sum(dim=1)
    cam = torch.relu(cam)

 
    cam_np = cam[0].cpu().numpy()
    cam_np = cv2.resize(cam_np, (224, 224))
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())  # normalize 0-1
    heatmap = cv2.applyColorMap((cam_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Convert original image back from tensor to numpy
    original = imgs[0].permute(1, 2, 0).cpu().detach().numpy()
    original = ((original * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])) * 255
    original = original.astype(np.uint8)
    
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    plt.axis('off')
    st.image(overlay,caption="Model Attention Heatmap", width=400)
    
col1,col2,col3 = st.columns([1,2,1])
if uploaded_files:
    with col2:
        grad_cam(model, img.squeeze(dim=0)) 
      
    st.warning("This Is A Project Please Treat It As Such, And Have A Real Medical Diagnosis In Case Of An Actual Lesion") 
    
with st.expander("ℹ️ How to interpret Grad-CAM"):
        st.markdown("""
        The heatmap shows which parts of the image the model focused on:
       
        - 🔴 **Red/Yellow areas**: Highest attention - key decision regions
        - 🟢 **Green areas**: Moderate attention
        - 🔵 **Blue/Purple areas**: Core focus region
        - ⚫ **Dark areas**: Low attention
       
        A good model should focus on the lesion itself, not background or artifacts.
        """)
        st.error("The model mainly attends to the lesion boundary and surrounding regions, while the lesion center shows lower activation.")
        
        
with tab2:
        samples_dir = "Samples"
    
        sample_definitions = {
            "Melanoma Sample 1": {
                "file": f"{samples_dir}/ISIC_0031709__1.jpg",
                "description": "Dark, irregular mole with asymmetric borders",
                "risk": "High Risk",
                "expected": "mel"
            },
            "Dermatofibroma": {
                "file": f"{samples_dir}/ISIC_0029177__6.jpg",
                "description": "Regular, benign (non-cancerous)",
                "risk": "Low Risk",
                "expected": "df"
            }
        }

        cols = st.columns(2)
        
        for idx, (name, info) in enumerate(sample_definitions.items()):
            with cols[idx]:
                st.markdown(f"**{name}**")
                st.caption(info['description'])
                st.markdown(f"{info['risk']}")
                
                
                if st.button(f"🔍 Classify This Sample", key=f"sample_{idx}"):
                    st.markdown("---")
                    st.markdown(f"### Analyzing: {name}")
                    st.image(info['file'])
                    df1 = Image.open(info['file'])
                    img = prediction(df1)
                    grad_cam(model, img.squeeze(dim=0)) 
                    
                    st.markdown("""
                        <div style='text-align: center; color: #666; padding: 20px;'>
                            <p><strong>Disclaimer:</strong> "This Is Taken From The TestSet And Is Not Trained By The Model!!"</p>
                        </div>
                    """, unsafe_allow_html=True)
                                      
                    
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Disclaimer:</strong> This is an educational project demonstrating AI-based skin lesion classification. 
        It is NOT a substitute for professional medical diagnosis. Always consult a qualified dermatologist for any skin concerns.</p>
    </div>
""", unsafe_allow_html=True)