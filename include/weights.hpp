#ifndef WEIGHTS_HPP
#define WEIGHTS_HPP

// 权重声明
extern float weights_conv1[6][1][5][5];
extern float weights_conv2[16][6][5][5];
extern float weights_fc1[120][16*5*5];
extern float weights_fc2[84][120];
extern float weights_fc3[10][84];

extern float bias_conv1[6];
extern float bias_conv2[16];
extern float bias_fc1[120];
extern float bias_fc2[84];
extern float bias_fc3[10];

/**
 * 加载模型参数
 * @return true表示成功，false表示失败
 */
bool load_weights();

#endif // WEIGHTS_HPP 