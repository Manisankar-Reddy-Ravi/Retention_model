def preprocess_input(df, encoders):
    df = df.copy()
    
    for col in ['Coverage_Type', 'Billing_Method', 'Payment_Method']:
        if col in df.columns:
            # Flatten any list to a scalar (e.g., ['Auto-Pay'] ➝ 'Auto-Pay')
            df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) else x)
            
            # Apply the saved LabelEncoder
            df[col] = encoders[col].transform(df[col])
    
    return df
