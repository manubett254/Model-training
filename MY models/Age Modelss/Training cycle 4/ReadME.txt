Columns in dataset: Index(['age_group', 'mfcc_0', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5',
       'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12',
       'mfcc_std_0', 'mfcc_std_1', 'mfcc_std_2', 'mfcc_std_3', 'mfcc_std_4',
       'mfcc_std_5', 'mfcc_std_6', 'mfcc_std_7', 'mfcc_std_8', 'mfcc_std_9',
       'mfcc_std_10', 'mfcc_std_11', 'mfcc_std_12', 'chroma_0', 'chroma_1',
       'chroma_2', 'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7',
       'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_std_0',
       'chroma_std_1', 'chroma_std_2', 'chroma_std_3', 'chroma_std_4',
       'chroma_std_5', 'chroma_std_6', 'chroma_std_7', 'chroma_std_8',
       'chroma_std_9', 'chroma_std_10', 'chroma_std_11', 'spec_contrast_0',
       'spec_contrast_1', 'spec_contrast_2', 'spec_contrast_3',
       'spec_contrast_4', 'spec_contrast_5', 'spec_contrast_6',
       'spec_contrast_std_0', 'spec_contrast_std_1', 'spec_contrast_std_2',
       'spec_contrast_std_3', 'spec_contrast_std_4', 'spec_contrast_std_5',
       'spec_contrast_std_6', 'zcr', 'zcr_std', 'rms', 'rms_std', 'centroid',
       'centroid_std', 'bandwidth', 'bandwidth_std', 'rolloff', 'rolloff_std',
       'hnr', 'hnr_std', 'pitch', 'pitch_std', 'gender'],
      dtype='object')
✅ Model Accuracy: 90.55%
🔍 Classification Report:
               precision    recall  f1-score   support

       adult       0.88      0.95      0.91       798
       child       0.98      0.98      0.98       665
       teens       0.27      0.10      0.14       103

    accuracy                           0.91      1566
   macro avg       0.71      0.67      0.68      1566
weighted avg       0.88      0.91      0.89      1566

🔍 Top 10 Important Features: [(np.float64(0.05259968050980521), 'spec_contrast_6'), (np.float64(0.046564281195371385), 'mfcc_std_3'), (np.float64(0.0376500223645422), 'mfcc_std_1'), (np.float64(0.02808511980176133), 'mfcc_4'), (np.float64(0.024655968542418315), 'mfcc_3'), (np.float64(0.02174937668038071), 'spec_contrast_1'), (np.float64(0.019968757005528564), 'gender'), (np.float64(0.019929077700123043), 'chroma_9'), (np.float64(0.019833876530839907), 'spec_contrast_3'), (np.float64(0.019462086470271697), 'spec_contrast_std_5')]
/usr/local/lib/python3.11/dist-packages/xgboost/core.py:158: UserWarning: [12:34:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "use_label_encoder" } are not used.

  warnings.warn(smsg, UserWarning)
✅ XGBoost Model Accuracy: 0.8984674329501916
✅ Model, Scaler, and LabelEncoder saved successfully!