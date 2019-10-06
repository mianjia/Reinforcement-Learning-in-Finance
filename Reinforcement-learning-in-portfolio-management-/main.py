# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import json
import pandas as pd
import numpy as np
import math
from decimal import Decimal
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from agents.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from agents.pg import PG
import datetime
import os
import seaborn as sns
sns.set_style("darkgrid")


eps=10e-8
epochs=0
M=0
PATH_prefix='' # link đến file csv

class StockTrader():
    # khởi đầu - stock trader sẽ tự chạy hàm reset()
    def __init__(self):
        self.reset()
        
    # reset lại các giá trị của stocktrader
    def reset(self):
        self.wealth = 10e3
        self.total_reward = 0
        self.ep_ave_max_q = 0
        self.loss = 0
        self.actor_loss=0

        self.wealth_history = []
        self.r_history = []
        self.w_history = []
        self.p_history = []
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(M)) # thêm actionnoise, M ở phía sau sẽ được khai báo bằng số lượng stock

    # update lại tổng kết cho stock trader, cập nhật các _history    
    def update_summary(self,loss,r,q_value,actor_loss,w,p):
        self.loss += loss
        self.actor_loss+=actor_loss
        self.total_reward+=r
        self.ep_ave_max_q += q_value
        self.r_history.append(r)
        self.wealth = self.wealth * math.exp(r)
        self.wealth_history.append(self.wealth)
        self.w_history.extend([','.join([str(Decimal(str(w0)).quantize(Decimal('0.00'))) for w0 in w.tolist()[0]])])
        self.p_history.extend([','.join([str(Decimal(str(p0)).quantize(Decimal('0.000'))) for p0 in p.tolist()])])

    # lưu xuống file csv  
    def write(self,codes,agent):
        global PATH_prefix
        wealth_history = pd.Series(self.wealth_history)
        r_history = pd.Series(self.r_history)
        w_history = pd.Series(self.w_history)
        p_history = pd.Series(self.p_history)
        history = pd.concat([wealth_history, r_history, w_history, p_history], axis=1)
        print('saving a file: '+PATH_prefix+agent+'-'.join(codes)+ '-' +str(math.exp(np.sum(self.r_history)) * 100) + '.csv')
        history.to_csv(PATH_prefix+agent + '-'.join(codes) + '-' + str(math.exp(np.sum(self.r_history)) * 100) + '.csv')

    #
    def print_result(self,epoch,agent,noise_flag):
        self.total_reward=math.exp(self.total_reward) * 100
        print('*-----Episode: {:d}, Reward:{:.6f}%-----*'.format(epoch, self.total_reward))        
        agent.write_summary(self.total_reward)
        agent.save_model()                       

    def plot_result(self):
        pd.Series(self.wealth_history).plot()
        plt.show()

    def action_processor(self,a,ratio):
        a = np.clip(a + self.noise() * ratio, 0, 1)
        a = a / (a.sum() + eps)
        return a

###########################################################################################################    
# RUN RL

# lấy các biến từ info của env
def parse_info(info):
    return info['reward'],info['continue'],info[ 'next state'],info['weight vector'],info ['price'],info['risk']

# luồng thông qua 
def traversal(stocktrader, 
              agent, 
              env, 
              epoch,
              noise_flag,
              framework,
              method,
              trainable):
    
    '''
    lấy thông tin-info của môi trường env
    r = reward, contin = thông tin của continue
    s = next state, w1 = weight vector
    p = price, risk = risk
    '''
    info = env.step(None,None,noise_flag)
    r,contin,s,w1,p,risk = parse_info(info) 
    
    contin=1 # continue
    t=0

    while contin: # while continue = 1 == True
        w2 = agent.predict(s,w1) # PREDICT

        env_info = env.step(w1, w2,noise_flag)   # lâý thông tin step tiếp theo từ môi trường
        r, contin, s_next, w1, p,risk = parse_info(env_info) # update các biến cho bước tiếp theo

        '''
        save transition với từng loại agent
        '''
        if framework=='PG':            
            agent.save_transition(s,p,w2,w1)            
        else:
            agent.save_transition(s, w2, r-risk, contin, s_next, w1)
        loss, q_value,actor_loss=0,0,0

        if framework=='DDPG':
            if not contin and trainable=="True":
                agent_info= agent.train(method,epoch)
                loss, q_value=agent_info["critic_loss"],agent_info["q_value"]
                if method=='model_based':
                    actor_loss=agent_info["actor_loss"]

        elif framework=='PPO':
            if not contin and trainable=="True":
                agent_info = agent.train(method, epoch)
                loss, q_value = agent_info["critic_loss"], agent_info["q_value"]
                if method=='model_based':
                    actor_loss=agent_info["actor_loss"]
        
        # train model PG khi mà contin = 0 và trainble = true
        # tức train lại PG với 
        elif framework=='PG':
            if not contin and trainable=="True":
                agent.train()

        stocktrader.update_summary(loss,r,q_value,actor_loss,w2,p)
        s = s_next
        t=t+1
        # loop tới khi nhận được tín hiệu contin = 0 từ môi trường
        
###########################################################################################################
# BACK TEST 
        
def maxdrawdown(arr):
    i = np.argmax((np.maximum.accumulate(arr) - arr)/np.maximum.accumulate(arr)) # end of the period
    j = np.argmax(arr[:i]) # start of period
    return (1-arr[i]/arr[j])


def backtest(agent,env,market):
    global PATH_prefix
    print("starting to backtest......")
    from agents.UCRP import UCRP
    from agents.Winner import WINNER
    from agents.Losser import LOSSER


    agents=[]
    agents.extend(agent)
    agents.append(WINNER())
    agents.append(UCRP())
    agents.append(LOSSER())
    labels=['PG','Winner','UCRP','Losser']

    wealths_result=[]
    rs_result=[]
    for i,agent in enumerate(agents):
        stocktrader = StockTrader()
        info = env.step(None, None,'False')
        r, contin, s, w1, p, risk = parse_info(info)
        contin = 1
        wealth=10000
        wealths = [wealth]
        rs=[1]
        while contin:
            w2 = agent.predict(s, w1)
            env_info = env.step(w1, w2,'False')
            r, contin, s_next, w1, p, risk = parse_info(env_info)
            wealth=wealth*math.exp(r)
            rs.append(math.exp(r)-1)
            wealths.append(wealth)
            s=s_next
            stocktrader.update_summary(0, r, 0, 0, w2, p)

        stocktrader.write(map(lambda x: str(x), env.get_codes()),labels[i])
        print('finish one agent')
        wealths_result.append(wealths)
        rs_result.append(rs)

#    print('资产名称','   ','平均日收益率','   ','夏普率','   ','最大回撤')
    print('Asset Name','   ','Average daily rate of return','   ','Sharp rate','   ','Maximum withdrawal')
    plt.figure(figsize=(8, 6), dpi=100)
    for i in range(len(agents)):
        plt.plot(wealths_result[i],label=labels[i])
        plt.title('RL - PM in the {} stock market'.format(market))
        mrr=float(np.mean(rs_result[i])*100)
        sharpe=float(np.mean(rs_result[i])/np.std(rs_result[i])*np.sqrt(252))
        maxdrawdown=float(max(1-min(wealths_result[i])/np.maximum.accumulate(wealths_result[i])))
        print(labels[i],'   ',round(mrr,3),'%','   ',round(sharpe,3),'  ',round(maxdrawdown,3))
    plt.legend()
    plt.savefig(PATH_prefix+'backtest.png')
    plt.show()

###########################################################################################################    
# KHAI BÁO CONFIG (MODEL INPUT PARAMETERS)

def parse_config(config,mode):
    codes = config["session"]["codes"]
    start_date = config["session"]["start_date"]
    end_date = config["session"]["end_date"]
    features = config["session"]["features"]
    agent_config = config["session"]["agents"]
    market = config["session"]["market_types"]
    noise_flag, record_flag, plot_flag=config["session"]["noise_flag"],config["session"]["record_flag"],config["session"]["plot_flag"]
    predictor, framework, window_length = agent_config
    reload_flag, trainable=config["session"]['reload_flag'],config["session"]['trainable']
    method=config["session"]['method']

    global epochs
    epochs = int(config["session"]["epochs"])

    if mode=='test':
        record_flag='True'
        noise_flag='False'
        plot_flag='True'
        reload_flag='True'
        trainable='False'
        method='model_free'

    print("*--------------------Training Status-------------------*")
    print("Date from",start_date,' to ',end_date)
    print('Features:',features)
    print("Agent:Noise(",noise_flag,')---Recoed(',noise_flag,')---Plot(',plot_flag,')')
    print("Market Type:",market)
    print("Stock codes:", codes)
    print("Predictor:",predictor,"  Framework:", framework,"  Window_length:",window_length)
    print("Epochs:",epochs)
    print("Trainable:",trainable)
    print("Reloaded Model:",reload_flag)
    print("Method",method)
    print("Noise_flag",noise_flag)
    print("Record_flag",record_flag)
    print("Plot_flag",plot_flag)


    return codes,start_date,end_date,features,agent_config,market,predictor,framework,window_length,noise_flag, record_flag, plot_flag,reload_flag,trainable,method



def session(config,args):
    global PATH_prefix
    from data.environment import Environment
    codes, start_date, end_date, features, agent_config, market,predictor, framework, window_length, noise_flag, record_flag, plot_flag,reload_flag,trainable,method = parse_config(config,args)
    env = Environment()

    global M
    if market == 'China':
        M=codes+1
    else:
        M=len(codes)+1
#     print("len codes",len(codes))
#     M=codes+1
    # M = số lượng stock -> ảnh huong đến noise - chi tiết from agents.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

    # if framework == 'DDPG':
    #     print("*-----------------Loading DDPG Agent---------------------*")
    #     from agents.ddpg import DDPG
    #     agent = DDPG(predictor, len(codes) + 1, int(window_length), len(features), '-'.join(agent_config), reload_flag,trainable)
    #
    # elif framework == 'PPO':
    #     print("*-----------------Loading PPO Agent---------------------*")
    #     from agents.ppo import PPO
    #     agent = PPO(predictor, len(codes) + 1, int(window_length), len(features), '-'.join(agent_config), reload_flag,trainable)


    stocktrader=StockTrader()
    PATH_prefix = "./result_new/PG/" + str(args['num']) + '/' #<-

    
    
    if args['mode']=='train':
        if not os.path.exists(PATH_prefix):
            print('Create new path at',PATH_prefix)
            os.makedirs(PATH_prefix)
            if market == "China":
                train_start_date, train_end_date, test_start_date, test_end_date, codes = env.get_repo(start_date, end_date, codes, market)
            else:
                train_start_date, train_end_date, test_start_date, test_end_date, codes = env.get_repo(start_date, end_date, len(codes), market)
            
            env.get_data(train_start_date, train_end_date, features, window_length, market, codes)
            
            print("Codes:", codes)
            print('Training Time Period:', train_start_date, '   ', train_end_date)
            print('Testing Time Period:', test_start_date, '   ', test_end_date)
            with open(PATH_prefix + 'config.json', 'w') as f:
                json.dump({"train_start_date": train_start_date.strftime('%Y-%m-%d'),
                           "train_end_date": train_end_date.strftime('%Y-%m-%d'),
                           "test_start_date": test_start_date.strftime('%Y-%m-%d'),
                           "test_end_date": test_end_date.strftime('%Y-%m-%d'), 
                           "codes": codes}, f)
                print("finish writing config")
                
        else:
            with open("./result_new/PG/" + str(args['num']) + '/config.json', 'r') as f:
                dict_data = json.load(f)
                print("successfully load config")
                
            if market == "China":
                train_start_date, train_end_date, test_start_date, test_end_date, codes = env.get_repo(start_date, end_date, codes, market)
            else:
                train_start_date, train_end_date, test_start_date, test_end_date, codes = env.get_repo(start_date, end_date, len(codes), market)
            
            env.get_data(train_start_date, train_end_date, features, window_length, market, codes)
                
#             train_start_date, train_end_date, codes = datetime.datetime.strptime(dict_data['train_start_date'],                                                                               '%Y-%m-%d'), datetime.datetime.strptime(dict_data['train_end_date'], '%Y-%m-%d'), dict_data['codes']
            
#             env.get_data(train_start_date, train_end_date, features, window_length, market, codes)

            
        for noise_flag in ['True']:#['False','True'] to train agents with noise and without noise in assets prices
            if framework == 'PG':
                print("*-----------------Loading PG Agent---------------------*")
                agent = PG(len(codes) + 1, int(window_length), len(features), '-'.join(agent_config), reload_flag, trainable,noise_flag,args['num'])

            print("Training with {:d}".format(epochs))
            for epoch in range(epochs):
                print("Now we are at epoch", epoch)
                traversal(stocktrader,agent,env,epoch,noise_flag,framework,method,trainable)
                if record_flag=='True':
                    stocktrader.write(epoch,framework)
                if plot_flag=='True':
                    stocktrader.plot_result()
                #print(agent)
                agent.reset_buffer()
                stocktrader.print_result(epoch,agent,noise_flag)
                stocktrader.reset()             
            agent.close()
            del agent

    #######
    # TESTING
    
    elif args['mode']=='test':
        with open("./result_new/PG/" + str(args['num']) + '/config.json', 'r') as f:
            dict_data=json.load(f)
        test_start_date, test_end_date, codes = datetime.datetime.strptime(dict_data['test_start_date'],'%Y-%m-%d'),datetime.datetime.strptime(dict_data['test_end_date'],'%Y-%m-%d'),dict_data['codes']
        env.get_data(test_start_date,test_end_date,features,window_length,market,codes)
        backtest([PG(len(codes) + 1, int(window_length), len(features), '-'.join(agent_config), 'True','False','True',args['num'])],
                 env,market)

        
        
def build_parser():
    parser = ArgumentParser(description='Provide arguments for training different DDPG or PPO models in Portfolio Management')
    parser.add_argument("--mode",choices=['train','test'], required=True)
    parser.add_argument("--num",type=int) # khai bao so thu tu cua model
    # args=vars(parser.parse_args())
    
    return parser


def main():
    parser = build_parser()
    args = vars(parser.parse_args())
    print(args)
    with open('config.json') as f:
        config=json.load(f)
        if args['mode']=='download':
            from data.download_data import DataDownloader
            data_downloader=DataDownloader(config)
            data_downloader.save_data()
        else:
            session(config,args)

if __name__=="__main__":
    main()