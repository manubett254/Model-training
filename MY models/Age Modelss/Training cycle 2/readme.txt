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
✅ Model Accuracy: 91.12%
🔍 Classification Report:
               precision    recall  f1-score   support

       adult       0.88      0.95      0.92       798
       child       0.98      0.98      0.98       665
       teens       0.30      0.10      0.15       103

    accuracy                           0.91      1566
   macro avg       0.72      0.68      0.68      1566
weighted avg       0.88      0.91      0.89      1566

✅ Model, Scaler, and LabelEncoder saved successfully!