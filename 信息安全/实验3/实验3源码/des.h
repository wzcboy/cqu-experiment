//
//  des.h
//  DES
//
//  Created by wzc on 2021/5/20.
//  Copyright © 2021 wzc. All rights reserved.
//

#ifndef des_h
#define des_h


#include <iostream>
#include <string>
#include <bitset>
#include <math.h>

// -----------------------生成子密钥过程--------------------
// 原密钥(2进制）
//int key[64] = {0,0,0,1,0,0,1,1, 0,0,1,1,0,1,0,0,
//    0,1,0,1,0,1,1,1, 0,1,1,1,1,0,0,1,
//    1,0,0,1,1,0,1,1, 1,0,1,1,1,1,0,0,
//    1,1,0,1,1,1,1,1, 1,1,1,1,0,0,0,1};
// 存储16个子密钥
int subKey[16][48];
// 记录子密钥左移次数
int leftMove[16] = { 1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1 };
// 17个子密钥的左半部分和右半部分
int C[17][28], D[17][28];
// 初始子密钥C0D0生成矩阵PC-1
// 64位原密钥 -> 56新密钥
int PC1[8][7] = {  {57,49,41,33,25,17,9},
    {1,58,50,42,34,26,18},
    {10,2,59,51,43,35,27},
    {19,11,3,60,52,44,36},
    {63,55,47,39,31,23,15},
    {7,62,54,46,38,30,22},
    {14,6,61,53,45,37,29},
    {21,13,5,28,20,12,4} };

// 子密钥生成矩阵PC-2
// 56位 -> 48位
int PC2[8][6] = {   {14,17,11,24,1,5},
    {3,28,15,6,21,10},
    {23,19,12,4,26,8 },
    {16,7,27,20,13,2},
    {41,52,31,37,47,55},
    {30,40,51,45,33,48},
    {44,49,39,56,34,53},
    {46,42,50,36,29,32} };

void calcSubKeys(int key[64]){
    // 先对原密钥进行PC-1的变换
    // 将新密钥拆分成C0D0
    int newKey[8][7];
    int l = 0, r = 0;
    for(int i=0;i<8;i++){
        for(int j=0;j<7;j++){
            int loc = PC1[i][j] - 1;
            newKey[i][j] = key[loc];
            if(l < 28) C[0][l++] = newKey[i][j];
            else D[0][r++] = newKey[i][j];
        }
    }
    
    // 产生16个子密钥
    for(int i=1;i<=16;i++){
        int times = leftMove[i-1];
        memcpy(C[i], C[i-1]+times, sizeof(int)*(28-times));
        memcpy(C[i]+28-times, C[i-1], sizeof(int)*times);
        memcpy(D[i], D[i-1]+times, sizeof(int)*(28-times));
        memcpy(D[i]+28-times, D[i-1], sizeof(int)*times);
    }
    // 56位->48位
    int CD[56];
    for(int i=1;i<=16;i++){
        memcpy(CD, C[i], sizeof(int)*28);
        memcpy(CD+28, D[i], sizeof(int)*28);
        for(int j=0;j<8;j++){
            for(int k=0;k<6;k++){
                int loc = PC2[j][k] - 1;
                subKey[i-1][j*6+k] = CD[loc];
            }
        }
    }
}

void intToBinArr(long long num, int res[64]){
    std::cout<<"密钥："<<num<<std::endl;
    memset(res, 0, sizeof(int)*64);
    int temp, i=63;
    while(num > 0){
        temp = num % 2;
        num = num / 2;
        res[i--] = temp;
    }
}
// -------------------- 加密过程 ---------------------
// 17个子密钥的左半部分和右半部分
int L[17][32], R[17][32];
// 密文初始置换矩阵
int IP[8][8] = {    {58,50,42,34,26,18,10,2,},
    {60,52,44,36,28,20,12,4,},
    {62,54,46,38,30,22,14,6,},
    {64,56,48,40,32,24,16,8,},
    {57,49,41,33,25,17,9,1,},
    {59,51,43,35,27,19,11,3,},
    {61,53,45,37,29,21,13,5,},
    {63,55,47,39,31,23,15,7}};
// 密文逆置换
int IP1[8][8] = {  {40,8,48,16,56,24,64,32},
    {39,7,47,15,55,23,63,31},
    {38,6,46,14,54,22,62,30},
    {37,5,45,13,53,21,61,29},
    {36,4,44,12,52,20,60,28},
    {35,3,43,11,51,19,59,27},
    {34,2,42,10,50,18,58,26},
    {33,1,41,9,49,17,57,25} };

// E扩展
int ETable[48] = {  32,1,2,3,4,5,
    4,5,6,7,8,9,
    8,9,10,11,12,13,
    12,13,14,15,16,17,
    16,17,18,19,20,21,
    20,21,22,23,24,25,
    24,25,26,27,28,29,
    28,29,30,31,32,1};

//p置换
int P[32] = {   16,7,20,21,29,12,28,17,
    1,15,23,26,5,18,31,10,
    2,8,24,14,32,27,3,9,
    19,13,30,6,22,11,4,25};

int boxS[8][4][16] = { //S1
    {   {14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7},
        {0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8},
        {4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0},
        {15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13}
    },
    //S2
    {   {15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10},
        {3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5},
        {0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15},
        {13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9}
    },
    //S3
    {   {10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8},
        {13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1},
        {13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7},
        {1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12}
    },
    //S4
    {   {7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15},
        {13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9},
        {10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4},
        {3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14}
    },
    //S5
    {   {2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9},
        {14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6},
        {4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14},
        {11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3}
    },
    //S6
    {   {12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11},
        {10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8},
        {9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6},
        {4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13}
    },
    //S7
    {   {4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1},
        {13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6},
        {1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2},
        {6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12}
    },
    //S8
    {   {13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7},
        {1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2},
        {7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8},
        {2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11}
    }
};

void initialPermutation(int M[64]){
    // 首先计算根据IP矩阵计算L0和R0
    int l = 0, r = 0;
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            int loc = IP[i][j] - 1;
            if(l < 32) L[0][l++] = M[loc];
            else R[0][r++] =  M[loc];
        }
    }
}

// 将32位数扩展为48位
int* E_func(int r[32]){
    int* res = new int[48];
    for(int i=0;i<48;i++){
        int loc = ETable[i] - 1;
        res[i] = r[loc];
    }
    
    return res;
}
// 将6位变成4位
int* S_func(int B[6], int boxID){
    int* res = new int[4];
    memset(res, 0, sizeof(int)*4);
    int row = B[0]*2 + B[5];
    int col = B[1]*8 + B[2]*4 + B[3]*2 + B[4];
    int val = boxS[boxID][row][col];
    // 将10进制转换成4位2进制数组
    int temp, i=3;
    while(val > 0){
        temp = val % 2;
        val = val / 2;
        res[i--] = temp;
    }
    return res;
}
// 首先将r扩展到48位，接着
int* f_func(int r[32], int k[48]){
    int* result = new int[32];
    // E(R)
    int* r_extend = E_func(r);
    // E(R) + K
    int xor_resut[48];
    for(int i=0;i<48;i++){
        xor_resut[i] = (r_extend[i]==k[i]) ? 0 : 1;
    }
    
    // 将xor_result分成8组
    int B[6];
    int* SB;
    for(int i=0;i<8;i++){
        memcpy(B, xor_resut+6*i, sizeof(int)*6);
        SB = S_func(B, i);
        memcpy(result+i*4, SB, sizeof(int)*4);
    }
    // 进行P变换
    int* PResult = new int[32];
    for(int i=0;i<32;i++){
        int loc = P[i] - 1;
        PResult[i] = result[loc];
    }
    //    delete[] result;
    return PResult;
}
// 计算所有的L和R: Ln = Rn-1 ; Rn = Ln-1 + f(Rn-1, Kn)
// 得到最后的密文
void calcCipher(int M[64]){
    initialPermutation(M);
    for(int i=1;i<=16;i++){
        memcpy(L[i], R[i-1], sizeof(int)*32);
        int* fResult = f_func(R[i-1], subKey[i-1]);
        for(int j=0;j<32;j++){
            R[i][j] = (fResult[j]==L[i-1][j]) ? 0 : 1;
        }
    }
}
// 加密逆过程（subkey反着用）
void calcPlain(int M[64]){
    initialPermutation(M);
    for(int i=1;i<=16;i++){
        memcpy(L[i], R[i-1], sizeof(int)*32);
        int* fResult = f_func(R[i-1], subKey[16-i]);
        for(int j=0;j<32;j++){
            R[i][j] = (fResult[j]==L[i-1][j]) ? 0 : 1;
        }
    }
}
// 对L16R16进行逆转
int* getFinalRes(){
    int* result = new int[64];
    int RL[64];
    memcpy(RL, R[16], sizeof(int)*32);
    memcpy(RL+32, L[16], sizeof(int)*32);
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            int loc = IP1[i][j]-1;
            result[i*8+j] = RL[loc];
        }
    }
    return result;
}

// 输入为二进制的字符串密文
// 打印16进制的字符串密文
void printCipher(std::string cipher){
    std::cout<<"0x";
    for(int i=0;i<cipher.length()/4;i++){
        int loc = i * 4;
        int hexNum = cipher[loc]*8 + cipher[loc+1]*4 + cipher[loc+2]*2 + cipher[loc+3];
        std::cout<<std::hex<<hexNum;
    }
    std::cout<<std::endl;
    
}
// 加密
std::string encode(char text[], int len){
    std::string res = "";
    // 每64位（8个字符）一组，进行加密
    // 循环次数
    int times = (len%8==0) ? len/8 : (len/8+1);
    int M[64];
    for(int n=0;n<times;n++){
        memset(M, 0, sizeof(M));
        for(int i=0;i<8;i++){
            int loc = n*8+i;
            // 当text字符长度不为8的整数倍时，后面填充为0
            if(loc >= len) break;
            std::bitset<8> bits =  std::bitset<8>(text[n*8+i]);
            // 翻转
            for(int i=0;i<4;i++){
                swap(bits[i], bits[7-i]);
            }
            for(int j=0;j<8;j++){
                M[i*8+j] = bits[j];
            }
        }
        
        calcCipher(M);
        int* cipher = getFinalRes();
        for(int i=0;i<64;i++){
            res += std::to_string(cipher[i]);
        }
    }
    return res;
}
// 解密
std::string decode(std::string cipher){
    int len = (int)cipher.length();
    std::string res = "";
    int* cipherArr = new int[len];
    for(int i=0;i<len;i++){
        cipherArr[i] = cipher[i] - '0';
    }
    // 一定是64的整数倍
    int eachCipher[64];
    for(int i=0;i<len/64;i++){
        memcpy(eachCipher, cipherArr+64*i, sizeof(int)*64);
        
        calcPlain(eachCipher);
        int* plain = getFinalRes();
        
        for(int i=0;i<8;i++){
            int sum = 0;
            for(int j=0;j<8;j++){
                sum += plain[i*8+j] * pow(2, 7-j);
            }
            res += (char)sum;
        }
    }
    
    return res;
}
#endif /* des_h */
