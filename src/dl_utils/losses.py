def dice_coef_loss(y_true, y_pred):
    y_t = K.flatten(y_true)
    y_p = K.flatten(y_pred)
    intersection = K.sum(y_t * y_p)
    return 1.0 - (2.0 * intersection + 1.0) / (K.sum(y_t) + K.sum(y_p) + 1.0)
