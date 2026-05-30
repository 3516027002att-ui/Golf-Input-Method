import os
import sys
import json
import argparse
from typing import List, Dict, Any

# 内置约 500 个极高频、无许可证争议的中文词汇（格式：词,全拼,简拼,默认词频）
DEFAULT_LEXICON_CSV = """的,de,d,9900
是,shi,s,9500
了,le,l,9000
在,zai,z,8500
我,wo,w,8000
你,ni,n,7900
他,ta,t,7000
她,ta,t,6500
它,ta,t,6000
们,men,m,6800
我们,women,wm,7800
你们,nimen,nm,7300
他们,tamen,tm,7100
有,you,y,6800
个,ge,g,6600
好,hao,h,6500
你好,nihao,nh,6700
你好啊,nihaoa,nha,4500
这,zhe,z,6400
要,yao,y,6300
想要,xiangyao,xy,6100
国,guo,g,6200
中国,zhongguo,zg,6500
人,ren,r,6100
和,he,h,6000
用,yong,y,5900
作,zuo,z,5800
时,shi,s,5700
去,qu,q,5600
来,lai,l,5500
会,hui,h,5400
能,neng,n,5300
对,dui,d,5200
都,dou,d,5100
多,duo,d,5000
少,shao,s,4500
多少,duoshao,ds,4700
没,mei,m,4400
没有,meiyou,my,4800
怎么,zenme,zm,4300
什么,shenme,sm,4600
为什么,weishenme,wsm,4200
觉得,juede,jd,4100
知道,zhidao,zd,4050
可以,keyi,ky,4000
现在,xianzai,xz,3900
今天,jintian,jt,3800
明天,mingtian,mt,3700
输入法,shurufa,srf,3600
自动排词,zidongpaici,zdpc,3500
模型,moxing,mx,3400
测试,ceshi,cs,3300
正确,zhengque,zq,3200
优秀,youxiu,yx,3100
框架,kuangjia,kj,3000
非常,feichang,fc,2900
开发,kaifa,kf,2800
代码,daima,dm,2700
程序,chengxu,cx,2600
系统,xitong,xt,2500
苹果,pingguo,pg,2400
香蕉,xiangjiao,xj,2300
西瓜,xigua,xg,2200
飞机,feiji,fj,2100
火车,huoche,hc,2000
汽车,qiche,qc,1900
天气,tianqi,tq,1800
下雨,xiayu,xy,1700
晴天,qingtian,qt,1600
高兴,gaoxing,gx,1500
快乐,kuaile,kl,1400
谢谢,xiexie,xx,1300
再见,zaijian,zj,1200
学习,xuexi,xx,1100
工作,gongzuo,gz,1000
生活,shenghuo,sh,900
时间,shijian,sj,800
地方,difang,df,700
朋友,pengyou,py,600
大家,dajia,dj,500
世界,shijie,sj,400
希望,xiwang,xw,300
成功,chenggong,cg,200
失败,shibai,sb,100
想,xiang,x,6500
一,yi,y,8000
二,er,e,5000
三,san,s,6800
四,si,s,4800
五,wu,w,5200
六,liu,l,4600
七,qi,q,4400
八,ba,b,4500
九,jiu,j,4200
十,shi,s,7000
百,bai,b,3800
千,qian,q,3600
万,wan,w,4100
亿,yi,y,2800
年,nian,n,6200
月,yue,y,5800
日,ri,r,5600
天,tian,t,6000
分,fen,f,4800
秒,miao,m,2500
点,dian,d,5900
时,shi,s,5700
分,fen,f,4800
大,da,d,7200
小,xiao,x,6800
长,chang,c,4100
短,duan,d,2900
高,gao,g,4300
矮,ai,a,1200
胖,pang,p,1400
瘦,shou,s,1300
新,xin,x,4800
旧,jiu,j,2600
老,lao,l,3500
年轻,nianqing,nq,3100
轻,qing,q,2800
重,zhong,z,3900
快,kuai,k,4400
慢,man,m,2700
坏,huai,h,2800
美,mei,m,4200
丑,chou,c,1100
远,yuan,y,3400
近,jin,j,3600
深,shen,s,2900
浅,qian,q,1800
硬,ying,y,2200
软,ruan,r,2100
电脑,diannao,dn,3800
手机,shouji,sj,4500
网络,wangluo,wl,4200
软件,ruanjian,rj,3600
硬件,yingjian,yj,2400
游戏,youxi,yx,4100
电影,dianying,dy,3900
音乐,yinyue,yy,3800
电视,dianshi,ds,3200
电话,dianhua,dh,3300
相机,xiangji,xj,1900
视频,shipin,sp,3950
图片,tupian,tp,3200
社会,shehui,sh,5100
国家,guojia,gj,5500
政府,zhengfu,zf,4400
人民,renmin,rm,4900
经济,jingji,jj,4800
政治,zhengzhi,zz,3800
文化,wenhua,wh,4600
艺术,yishu,ys,3200
科学,kexue,kx,4100
技术,jishu,js,4500
历史,lishi,ls,4200
地理,dili,dl,2200
数学,shuxue,sx,3100
物理,wuli,wl,2900
化学,huaxue,hx,2600
生物,shengwu,sw,2800
健康,jiankang,jk,4600
医院,yiyuan,yy,3500
医生,yisheng,ys,3600
药物,yaowu,yw,2500
体育,tiyu,ty,3400
运动,yundong,yd,3900
足球,zuqiu,zq,2800
篮球,lanqiu,lq,2600
跑步,paobu,pb,2300
游泳,youyong,yy,2200
旅游,lvyou,ly,3600
旅行,lvxing,lx,3100
风景,fengjing,fj,2900
照片,zhaopian,zp,3200
地图,ditu,dt,2400
学校,xuexiao,xx,4300
大学,daxue,dx,3900
教师,jiaoshi,js,2900
学生,xuesheng,xs,4200
教室,jiaoshi,js,2300
考试,kaoshi,ks,3200
知识,zhishi,zs,3900
问题,wenti,wt,5300
答案,daan,da,3300
方法,fangfa,ff,4700
方案,fangan,fa,3200
计划,jihua,jh,3900
目标,mubiao,mb,4100
任务,renwu,rw,3800
过程,guocheng,gc,4200
结果,jieguo,jg,4600
原因,yuanyin,yy,3700
影响,yingxiang,yx,4100
关系,guanxi,gx,4300
合作,hezuo,hz,3900
交流,jiaoliu,jl,3600
讨论,taolun,tl,3100
会议,huiyi,hy,3300
公司,gongsi,gs,4900
企业,qiye,qy,4500
市场,shichang,sc,4600
产品,chanpin,cp,4400
服务,fuwu,fw,4700
品牌,pinpai,pp,3200
广告,guanggao,gg,3100
价格,jiage,jg,3900
成本,chengben,cb,3400
利润,lirun,lr,2800
投资,touzi,tz,3800
资金,zijin,zj,3300
银行,yinhang,yh,3900
信用卡,xinyongka,xyk,2500
消费,xiaofei,xf,3600
贸易,maoyi,my,2900
商业,shangye,sy,3800
工业,gongye,gy,3300
农业,nongye,ny,3100
林业,linye,ly,1400
渔业,yuye,yy,1500
城市,chengshi,cs,4300
农村,nongcun,nc,3300
街道,jiedao,jd,2500
建筑,jianzhu,jz,3200
公园,gongyuan,gy,2900
图书馆,tushuguan,tsg,2500
博物馆,bowuguan,bwg,2100
商店,shangdian,sd,2800
超市,chaoshi,cs,3100
餐厅,canting,ct,2700
咖啡馆,kafeiguan,kfg,2100
酒店,jiudian,jd,3200
住房,zhufang,zf,3400
公寓,gongyu,gy,2300
家庭,jiating,jt,4500
父母,fumu,fm,3600
孩子,haizi,hz,4100
兄弟,xiongdi,xd,2600
姐妹,jiemei,jm,2400
爷爷,yeye,yy,2100
奶奶,nainai,nn,2000
结婚,jiehun,jh,2900
恋爱,lianai,la,2500
幸福,xingfu,xf,3900
痛苦,tongku,tk,2600
悲伤,beishang,bs,2100
愤怒,fennu,fn,2200
害怕,haipa,hp,2700
惊讶,jingya,jy,2000
怀疑,huaiyi,hy,2900
相信,xiangxin,xx,4100
理解,lijie,lj,4200
支持,zhichi,zc,3900
反对,fandui,fd,3200
讨论,taolun,tl,3100
研究,yanjiu,yj,4300
分析,fenxi,fx,4100
设计,sheji,sj,3900
制造,zhizao,zz,3200
销售,xiaoshou,xs,3400
购买,goumai,gm,2900
支付,zhifu,zf,3300
安全,anquan,aq,4500
危险,weixian,wx,2800
紧急,jinji,jj,2500
救助,jiuzhu,jz,2300
法律,falu,fl,3900
规则,guize,gz,3400
权利,quanli,ql,3600
义务,yiwu,yw,2900
道德,daode,dd,3200
责任,zeren,zr,3900
信用,xinyong,xy,3300
诚实,chengshi,cs,2800
勇敢,yonggan,yg,2300
善良,shanliang,sl,2600
聪明,congming,cm,2700
勤奋,qinfen,qf,2200
懒惰,landuo,ld,1300
自私,zisi,zs,1600
无私,wusi,ws,1500
自由,ziyou,zy,3800
平等,pingdeng,pd,3400
公正,gongzheng,gz,3200
和平,heping,hp,3500
战争,zhanzheng,zz,3300
武器,wuqi,wq,2400
军队,jundui,jd,3100
警察,jingcha,jc,3200
罪犯,zuifan,zf,2300
监狱,jianyu,jy,1700
法庭,fating,ft,2200
律师,lvshi,ls,2800
法官,faguan,fg,2500
原告,yuangao,yg,1300
被告,beigao,bg,1500
证人,zhengren,zr,1900
证据,zhengju,zj,3200
事实,shishi,ss,3900
真理,zhenli,zl,2800
谎言,huangyan,hy,2100
错误,cuowu,cw,3400
正确,zhengque,zq,3200
标准,biaozhun,bz,3800
水平,shuiping,sp,3900
质量,zhiliang,zl,4100
数量,shuliang,sl,3600
比例,bili,bl,2900
结构,jiegou,jg,3800
形式,xingshi,xs,3600
内容,neirong,nr,3900
本质,benzhi,bz,3200
现象,xianxiang,xx,3400
规律,guilv,gl,3100
趋势,qushi,qs,2900
变化,bianhua,bh,4200
发展,fazhan,fz,4800
创新,chuangxin,cx,3900
改革,gaige,gg,3600
开放,kaifang,kf,3400
封闭,fengbi,fb,1900
保守,baoshou,bs,2100
进步,jinbu,jb,3200
退步,tuibu,tb,1300
稳定,wending,wd,3800
动荡,dongdang,dd,1900
危机,weiji,wj,3400
机遇,jiyu,jy,2900
挑战,tiaozhan,tz,3300
未来,weilai,wl,4300
过去,guoqu,gq,3900
当前,dangqian,dq,3800
临时,linshi,ls,2400
永久,yongjiu,yj,2300
瞬间,shunjian,sj,2200
永恒,yongheng,yh,2100
宇宙,yuzhou,yz,3200
地球,diqiu,dq,3600
太阳,taiyang,ty,3500
月亮,yueliang,yl,2900
星星,xingxing,xx,2600
天空,tiankong,tk,3100
海洋,haiyang,hy,3300
陆地,ludi,ld,2500
森林,senlin,sl,2900
沙漠,shamo,sm,1800
草原,caoyuan,cy,2200
河流,heliu,hl,2600
湖泊,hubo,hb,1900
山脉,shanmai,sm,2100
火山,huoshan,hs,1800
地震,dizhen,dz,2400
海啸,haixiao,hx,1500
台风,taifeng,tf,2100
暴风雨,baofengyu,bfy,1900
闪电,shandian,sd,1700
打雷,dalei,dl,1400
彩虹,caihong,ch,1800
空气,kongqi,kq,3300
水分,shuifen,sf,2500
土壤,turang,tr,2400
矿物,kuangwu,kw,2200
金属,jinshu,js,2900
黄金,huangjin,hj,3100
白银,baiyin,by,1900
钢铁,gangtie,gt,2600
煤炭,meitan,mt,2300
石油,shiyou,sy,3200
天然气,tianranqi,trq,2900
核能,heneng,hn,2100
太阳能,taiyangneng,tyn,2600
风能,fengneng,fn,2200
水能,shuineng,sn,1900
污染,wuran,wr,3100
环保,huanbao,hb,3600
生态,shengtai,st,3400
气候,qihou,qh,3100
气温,qiwen,qw,2500
湿度,shidu,sd,1800
风向,fengxiang,fx,1600
风力,fengli,fl,1900
降雨量,jiangyuliang,jyl,2100
降雪量,jiangxueliang,jxl,1400
四季,siji,sj,2500
春天,chuntian,ct,3300
夏天,xiatian,xt,3100
秋天,qiutian,qt,3200
冬天,dongtian,dt,2900
温暖,wennuan,wn,2900
炎热,yanre,yr,2100
凉爽,liangshuang,ls,2400
寒冷,hanleng,hl,2600
潮湿,chaoshi,cs,2100
干燥,ganzao,gz,2200
健康,jiankang,jk,4600
疾病,jibing,jb,3300
感冒,ganmao,gm,2400
发烧,fashao,fs,2100
咳嗽,kesou,ks,1900
头痛,toutong,tt,1800
胃痛,weitong,wt,1500
牙痛,yatong,yt,1100
心脑血管,xinnaoxueguan,xnxg,1400
癌症,aizheng,az,2600
糖尿病,tangniaobing,tnb,2100
高血压,gaoxueya,gxy,2300
肥胖,feipang,fp,2200
近视,jinshi,js,1900
预防,yufang,yf,3200
治疗,zhiliao,zl,3500
康复,kangfu,kf,2600
养生,yangsheng,ys,2900
饮食,yinshi,ys,3600
营养,yingyang,yy,3400
蛋白质,danbaizhi,dbz,2800
脂肪,zhifang,zf,2300
维生素,weishengsu,wss,2900
矿物质,kuangwuzhi,kwz,2100
膳食纤维,shanshangxianwei,ssxw,1900
水,shui,s,6800
牛奶,niunai,nn,3200
鸡蛋,jidan,jd,3300
面包,mianbao,mb,2900
米饭,mifan,mf,3100
面条,miantiao,mt,2500
蔬菜,shucai,sc,3400
水果,shuiguo,sg,3600
肉类,roulei,rl,2900
鱼类,yulei,yl,2600
豆制品,douzhipin,dzp,2100
坚果,jianguo,jg,2300
食用油,shiyongyou,syy,2200
调味品,tiaoweipin,twp,2100
茶,cha,c,3300
咖啡,kafei,kf,3200
果汁,guozhi,gz,2400
汽水,qishui,qs,1900
啤酒,pijiu,pj,2600
白酒,baijiu,bj,2100
红酒,hongjiu,hj,2200
饮用水,yinyongshui,yys,2500
"""

def parse_csv_lexicon(csv_text: str) -> List[Dict[str, Any]]:
    entries = []
    for line in csv_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) == 4:
            word, pinyin, short_pinyin, freq_str = parts
            try:
                freq = float(freq_str)
                entries.append({
                    "word": word,
                    "pinyin": pinyin,
                    "short_pinyin": short_pinyin,
                    "freq": freq,
                    "source": "lexicon"
                })
            except ValueError:
                pass
    return entries

def main():
    parser = argparse.ArgumentParser(description="golf 输入表扩充与导入脚本")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="外部待导入的词库文本文件。格式：每行一个词(词 拼音 简拼 词频)，以逗号或空格分隔。如未指定，将默认导入内置约 500 个常用词。"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.join("data", "lexicon", "dict.jsonl"),
        help="输出的 jsonl 词典文件路径，默认为 data/lexicon/dict.jsonl"
    )
    parser.add_argument(
        "--append", "-a",
        action="store_true",
        help="是否以追加模式合并到现有词典中，默认覆盖"
    )
    args = parser.parse_args()

    # 1. 准备要导入的条目
    imported_entries = []

    if args.input:
        if not os.path.exists(args.input):
            print(f"错误: 输入文件不存在: {args.input}")
            sys.exit(1)
        print(f"正在从外部文件 {args.input} 导入词表...")
        with open(args.input, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                # 支持逗号、Tab、空格分隔
                delimiters = [",", "\t", " "]
                parts = []
                for d in delimiters:
                    if d in line:
                        parts = [p.strip() for p in line.split(d) if p.strip()]
                        break
                if not parts:
                    parts = [line]

                if len(parts) >= 2:
                    word = parts[0]
                    pinyin = parts[1]
                    short_pinyin = parts[2] if len(parts) >= 3 else "".join([ch[0] for ch in pinyin.split() if ch])
                    freq = 1000.0
                    if len(parts) >= 4:
                        try:
                            freq = float(parts[3])
                        except ValueError:
                            pass
                    imported_entries.append({
                        "word": word,
                        "pinyin": pinyin,
                        "short_pinyin": short_pinyin,
                        "freq": freq,
                        "source": "imported"
                    })
                else:
                    print(f"警告: 忽略第 {idx} 行非法格式: {line}")
    else:
        print("未指定外部文件，将导入内置的 ~500 个高频极简词库...")
        imported_entries = parse_csv_lexicon(DEFAULT_LEXICON_CSV)

    if not imported_entries:
        print("未提取到任何词库条目，导入终止。")
        sys.exit(1)

    # 2. 如果是追加模式，先读取现有词表做去重合并
    final_entries = []
    existing_keys = set()

    if args.append and os.path.exists(args.output):
        print(f"读取现有词表 {args.output} 以进行追加合并...")
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    key = (data["word"], data["pinyin"])
                    if key not in existing_keys:
                        existing_keys.add(key)
                        final_entries.append(data)
        except Exception as e:
            print(f"警告: 读取现有词表失败 ({e})，将重写新词表。")

    # 合并新条目
    added_count = 0
    for entry in imported_entries:
        key = (entry["word"], entry["pinyin"])
        if key not in existing_keys:
            existing_keys.add(key)
            final_entries.append(entry)
            added_count += 1
        else:
            # 如果已存在，更新词频为较大者
            for ex in final_entries:
                if ex["word"] == entry["word"] and ex["pinyin"] == entry["pinyin"]:
                    ex["freq"] = max(ex["freq"], entry["freq"])
                    break

    # 按拼音排序，同拼音下按词频降序，使加载更规范
    final_entries.sort(key=lambda x: (x["pinyin"], -x["freq"]))

    # 3. 写入输出文件
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            for entry in final_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"成功导出词表到: {args.output}")
        print(f"本次新增/合并词条: {added_count} 个，总计有效词条数: {len(final_entries)} 个。")
    except Exception as e:
        print(f"错误: 写入输出文件失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
