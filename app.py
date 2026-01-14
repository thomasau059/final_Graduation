import os
import time
import secrets
import numpy as np
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import json

print("🍉 西瓜分類 Web 應用系統 🍉")
print("="*50)

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'jfif'}
MAX_FILE_SIZE = 10 * 1024 * 1024

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# 載入模型
img_width, img_height = 224, 224
model_path = './models/watermelon_model.h5'
class_indices_path = './models/class_indices.txt'

try:
    model = load_model(model_path)
    print("✅ 模型載入成功")
except Exception as e:
    print(f"❌ 模型載入錯誤: {e}")
    model = None

# 載入類別
try:
    with open(class_indices_path, 'r', encoding='utf-8') as f:
        class_indices_str = f.read().replace("'", '"')
        class_indices = json.loads(class_indices_str)
    CLASS_NAMES = {v: k for k, v in class_indices.items()}
    print(f"✅ 類別名稱: {CLASS_NAMES}")
except Exception as e:
    CLASS_NAMES = {0: 'good', 1: 'not_good', 2: 'others'}
    print(f"⚠ 使用預設類別名稱 ({e})")

# 閾值設定
THRESHOLDS = {
    'good': 0.60,
    'not_good': 0.60,
    'others': 0.50,
    'uncertain': 0.60
}

def allowed_file(filename):
    """檢查文件格式是否允許"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_filename(extension):
    """生成安全的隨機文件名"""
    return f"{secrets.token_hex(8)}.{extension}"

def predict_image(file_path):
    """預測圖片類別"""
    if model is None:
        return None
    
    try:
        img = load_img(file_path, target_size=(img_height, img_width))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)
        probs = predictions[0]
        class_idx = np.argmax(probs)
        confidence = probs[class_idx]
        class_name = CLASS_NAMES.get(class_idx, 'unknown')
        
        return {
            'class_index': int(class_idx),
            'class_name': class_name,
            'confidence': float(confidence),
            'probabilities': [float(p) for p in probs],
            'file_path': file_path,
            'is_watermelon': class_name in ['good', 'not_good'],
            'is_others': class_name == 'others',
            'is_confident': confidence >= THRESHOLDS.get(class_name, 0.6)
        }
    except Exception as e:
        print(f"預測錯誤: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def main_handler():
    if request.method == 'GET':
        return render_template('index.html',
                               label='請上傳西瓜圖片進行分析...',
                               imagesource=None)
    
    start_time = time.time()
    
    # 檢查文件是否存在
    if 'file' not in request.files:
        return render_template('index.html',
                               label='未選擇文件',
                               imagesource=None)
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html',
                               label='文件名稱為空',
                               imagesource=None)
    
    if not allowed_file(file.filename):
        return render_template('index.html',
                               label='不支持的文件格式 (僅限 .jpg, .jpeg, .png)',
                               imagesource=None)
    
    try:
        # 保存文件
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        new_filename = generate_filename(ext)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        file.save(file_path)
        
        # 預測
        result = predict_image(file_path)
        
        if result is None:
            return render_template('error.html',
                                   error_type='model_error',
                                   message='系統錯誤：分析圖片失敗')
        
        # 情況 1: 其他物體
        if result['is_others']:
            if result['confidence'] >= THRESHOLDS['others']:
                return redirect(url_for('error_page',
                                        error_type='not_watermelon',
                                        confidence=result['confidence'],
                                        filename=new_filename))
            else:
                return redirect(url_for('uncertain_page',
                                        confidence=result['confidence'],
                                        filename=new_filename))
        
        # 情況 2: 信心度不足
        elif not result['is_confident']:
            return redirect(url_for('uncertain_page',
                                    confidence=result['confidence'],
                                    class_name=result['class_name'],
                                    filename=new_filename))
        
        # 情況 3: 西瓜且信心度高
        else:
            class_chinese = {
                'good': '🍉 好西瓜',
                'not_good': '🍉 不好西瓜'
            }
            
            label_text = f"{class_chinese.get(result['class_name'])} ({result['confidence']*100:.1f}%)"
            
            return render_template('index.html',
                                   label=label_text,
                                   imagesource=f'../uploads/{new_filename}',
                                   prob_good=result['probabilities'][0]*100,
                                   prob_not_good=result['probabilities'][1]*100,
                                   prob_others=result['probabilities'][2]*100,
                                   processing_time=round(time.time() - start_time, 2))
    
    except Exception as e:
        return render_template('error.html',
                               error_type='processing_error',
                               message=f'處理錯誤: {str(e)}')

@app.route('/error')
def error_page():
    error_type = request.args.get('error_type', 'unknown')
    confidence = float(request.args.get('confidence', 0))
    filename = request.args.get('filename', '')
    
    error_messages = {
        'not_watermelon': {
            'title': '⚠️ 不是西瓜',
            'message': '系統確定這不是西瓜。',
            'details': f'信心度: {confidence*100:.1f}%',
            'solution': '請上傳真正的西瓜圖片。'
        },
        'model_error': {
            'title': '🔧 系統錯誤',
            'message': '分析圖片時發生錯誤。',
            'solution': '請重試或聯繫管理員。'
        },
        'processing_error': {
            'title': '🔄 處理錯誤',
            'message': '無法處理此圖片。',
            'solution': '請嘗試其他圖片。'
        }
    }
    
    error_info = error_messages.get(error_type, {
        'title': '❓ 未知錯誤',
        'message': '發生未知錯誤。',
        'solution': '請稍後重試。'
    })
    
    return render_template('error.html',
                           error_title=error_info['title'],
                           error_message=error_info['message'],
                           error_details=error_info.get('details', ''),
                           error_solution=error_info['solution'],
                           imagesource=f'../uploads/{filename}' if filename else None)

@app.route('/uncertain')
def uncertain_page():
    confidence = float(request.args.get('confidence', 0))
    class_name = request.args.get('class_name', 'unknown')
    filename = request.args.get('filename', '')
    
    class_display = {
        'good': '🍉 好西瓜',
        'not_good': '🍉 不好西瓜',
        'others': '❓ 其他物體',
        'unknown': '❓ 未知'
    }
    
    return render_template('uncertain.html',
                           confidence=confidence*100,
                           predicted_class=class_display.get(class_name),
                           imagesource=f'../uploads/{filename}' if filename else None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("="*50)
    print("系統已啟動")
    print(f"模型狀態: {'✅ 已載入' if model else '❌ 未載入'}")
    print(f"類別: {CLASS_NAMES}")
    print(f"上傳目錄: {UPLOAD_FOLDER}")
    print("="*50)
    print("請訪問: http://localhost:3000")
    print("="*50)
    
    app.run(host='0.0.0.0', port=3000, debug=False)