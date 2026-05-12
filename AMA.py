import numpy as np
import pandas as pd


def calculate_ama(df, bos_p=20, slow_p=40, fast_p=6, push_fac=0.5, anch_w=0.1, smth_p=3, filter_th=0.0001):
    """
    Calculate the Adaptive Moving Average (AMA) and its signals.
    
    The AMA adapts its smoothing factor based on market impulse (Break of Structure).
    It includes a noise filter and a push factor based on volatility.
    
    Parameters:
        df (pd.DataFrame): Dataframe with 'Close', 'High', 'Low'.
        bos_p (int): Lookback for Break of Structure (BOS).
        slow_p (int): Smoothing factor for the slow period.
        fast_p (int): Smoothing factor for the fast period.
        push_fac (float): Volatility adjustment factor.
        anch_w (float): Anchor weight for midline attraction.
        smth_p (int): Final EMA smoothing period.
        filter_th (float): Minimum price change threshold for signal generation.
        
    Returns:
        pd.DataFrame: Original dataframe with 'AMA', 'ATR', and 'AMA_Signal' columns.
    """
    df = df.copy()
    
    # Pre-calculate inputs
    c = df['Close'].values
    h = df['High'].values
    l = df['Low'].values
    
    length = len(df)
    
    # 1. Base impulse signal
    bos_imp = np.zeros(length)
    for i in range(1, length):
        if c[i] > h[i-1]:
            bos_imp[i] = 1
        elif c[i] < l[i-1]:
            bos_imp[i] = -1
            
    # 2. True Range and ATR
    tr_val = np.zeros(length)
    for i in range(1, length):
        tr_val[i] = max(h[i] - l[i], abs(c[i-1] - h[i]), abs(c[i-1] - l[i]))
    
    # ATR is a simple moving average of TR
    atr_val = pd.Series(tr_val).rolling(window=14, min_periods=1).mean().values
    
    # 3. Midline
    hhv_20 = pd.Series(h).shift(1).rolling(window=20, min_periods=1).max().values
    llv_20 = pd.Series(l).shift(1).rolling(window=20, min_periods=1).min().values
    midline = (hhv_20 + llv_20) / 2.0
    
    # Smoothing factors
    a_slow = 2.0 / (slow_p + 1.0)
    a_fast = 2.0 / (fast_p + 1.0)
    ema_alpha = 2.0 / (smth_p + 1.0)
    
    ama_list = np.zeros(length)
    
    bos_act = 0.0
    bos_bias = 0.0
    raw_main = 0.0
    ama_val = 0.0
    initialized = False
    
    for i in range(length):
        if i == 0:
            bos_act = abs(bos_imp[i])
            bos_bias = bos_imp[i]
        else:
            bos_act = bos_act + (abs(bos_imp[i]) - bos_act) / bos_p
            bos_bias = bos_bias + (bos_imp[i] - bos_bias) / bos_p
            
        a_base = a_slow + (a_fast - a_slow) * bos_act
        
        curr_atr = atr_val[i]
        curr_mid = midline[i]
        curr_c = c[i]
        
        if not np.isnan(curr_atr) and not np.isnan(curr_mid):
            tgt = curr_c + push_fac * bos_bias * curr_atr + anch_w * (curr_mid - curr_c)
            
            if not initialized:
                raw_main = tgt
                ama_val = raw_main
                initialized = True
            else:
                raw_main = a_base * tgt + (1.0 - a_base) * raw_main
                ama_val = ama_val + ema_alpha * (raw_main - ama_val)
                
        ama_list[i] = ama_val
        
    df['AMA'] = ama_list
    df['ATR'] = atr_val
    
    # 4. Quantitative Signals
    # AMA signal threshold 0.01%
    is_up_thresh = df['AMA'] > df['AMA'].shift(1) * (1.0 + filter_th)
    was_up_thresh = is_up_thresh.shift(1).fillna(False)
    
    buy_cond = is_up_thresh & (~was_up_thresh)
    sell_cond = (~is_up_thresh) & was_up_thresh
    
    # 1 for Buy, -1 for Sell, 0 for None
    signals = np.zeros(length)
    signals[buy_cond] = 1
    signals[sell_cond] = -1
    
    df['AMA_Signal'] = signals
    
    return df