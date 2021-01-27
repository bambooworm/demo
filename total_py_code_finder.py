#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:23:05 2018

@author: shenchen
"""
import re
import os
import time
import glob2

#高亮显示
def HLW(c, color='w', end=''):
    """
    30-39：字体颜色
        30: 黑 31: 红 32: 绿 33: 黄 34: 蓝 35: 紫 36: 绿 37: 白
    """
    if color == 'w':
        high_light_word(c, 37, end=end)
    elif color == 'r':
        high_light_word(c, 31, end=end)
    elif color == 'g':
        high_light_word(c, 32, end=end)
    elif color == 'b':
        high_light_word(c, 34, end=end)
    elif color == 'y':
        high_light_word(c, 33, end=end)
    elif color == 'p':
        high_light_word(c, 35, end=end)
    else:
        high_light_word(c, end=end)

#高亮显示
def high_light_word(c, color=37, end=''):
    print('\033[1;'+str(color)+'m',end='')
    print(str(c),end='')
    print('\033[0m',end=end)


#记录一个文件的行代码
def loadLine(fpath):
    count = 0
    codes = []
    try:
        for file_line in open(fpath, encoding='utf-8').readlines():
            codes.append(file_line)
            count += 1
    except:
        for file_line in open(fpath, encoding='cp852').readlines():
            codes.append(file_line)
            count += 1
    return codes, count


#匹配自定义的代码
def pairing(filelists, my_code, flag=0):

    paired = []
    cores = []
    foos = []

    for fpath in filelists:

        do_paired=False
        fcodes, fcount = loadLine(fpath)

        for i,fcode in enumerate(fcodes):


            pairs=re.finditer(my_code, fcode, flag)


            # ========================================
            if 'from ' in fcode:
#                core = fcode.split('from ')[-1].split(' import')[0].split('.')[0]
                cores.append(fcode.strip())
            elif 'import ' in fcode:
#                core = fcode.split('import ')[-1].split('.')[0]
                cores.append(fcode.strip())

            # ========================================
            if fcode[:4] == 'def ':
                foo = fcode.split('def ')[-1].split('(')[0]
                foo = 'from .{} import {}'.format(os.path.basename(fpath), foo.strip())
                foos.append(foo)





            t = 0
            do_muilt_pair = False
            for k,pair in enumerate(pairs):
                if not do_paired:
                    HLW('"'+fpath+'"', 'g', end='\n')
                    print()
                    do_paired=True

                if not do_muilt_pair:
                    print('line: {} '.format(i).ljust(15), end='')

                s,e = pair.span()
                if not do_muilt_pair:
                    print(fcode[t:s].lstrip(),end='')
                    do_muilt_pair = True
                else:
                    print(fcode[t:s],end='')

                HLW(fcode[s:e])
                paired.append(fcode[s:e])
                t = e

            if do_muilt_pair:
                print(fcode[e:].rstrip(),end='')
                print('')

        if do_paired:
            print()
            print('-'*56)

    return sorted(paired), sorted(list(set(cores))), sorted(list(set(foos)))

if __name__ == '__main__' :
    HLW('='*56,'b')
    print()

    startTime = time.perf_counter()
    basedir = 'F:\\Workplace\\ml-sound-classifier-master\\**\\*.py' # 需要搜索的

    """
    正则表达：
        \s*
            多个空格（或0个空格）的匹配
        {[\s\S]*?}
            多个{}的匹配，{}内任意长度内容（或无内容）
        [0-9]{...}
            多个数字串的匹配
            {...} 内填写匹配的数字串长度，
            可以是固定长度，例如[0-9]{3} ；也可是范围长度，例如[0-9]{3-9}
    """
#    my_code = "Errno\s*?[0-9]{3}" # 需要搜索的代码，可以是正则表达
#    my_code = "{[\s\S]*?}"
#    my_code = "assert[\s\S]*?isType"
    my_code = "visualize_cam_audio"

    filelists=glob2.glob(basedir)
    """
    flag : 可选，表示匹配模式，比如忽略大小写，多行模式等，具体参数为：
        re.I 忽略大小写
        re.L 表示特殊字符集 \w, \W, \b, \B, \s, \S 依赖于当前环境
        re.M 多行模式
        re.S 即为 . 并且包括换行符在内的任意字符（. 不包括换行符）
        re.U 表示特殊字符集 \w, \W, \b, \B, \d, \D, \s, \S 依赖于 Unicode 字符属性数据库
        re.X 为了增加可读性，忽略空格和 # 后面的注释
    """
    res,cores,foos = pairing(filelists, my_code, flag=0)


