        for i in self.hero_agents_list:
            self.hero_agents_list[i].model.init_path()
        
        with torch.no_grad():
            # sample `worker_steps` from each worker
            #there are no worker steps... rather there are full episodes

            step = 0
            fault_condition = False
            clear_output(wait=True)
            phase = 0
            
        



            

            
            for episode in range(self.num_episodes):#num_episodes):
                
                total_rewards = {i:0 for i in env.possible_agents} #i can report this
                trace = tensor.zeros((self.context_len,self.qnet_config_dict['ob_space']))
                action=1
                #return2g = 110
                
                

                for i in self.hero_agents_list:
                    self.hero_agents_list[i].model.init_CL_sample_store()





                
                
                if fault_condition:
                    env = env_risk(**(self.env_config  #| {"render_mode" : None,"bad_mov_penalization" : 0.01,"render_":False#False
                                                        # }
                                     )
                                      )#game.env(render_mode=None)

                curren_epi = episode + (iteration-1)*self.num_episodes
                env.reset(curren_epi) #for riplication
                
                fault_condition = False
                step_count = 0
                
                self.reset_moves_hero_agents()
                is_draw = 0
                
                #draw_territory_count = 0
                #is_third = 0

                for agent in env.agent_iter():
                    e_t = env.terminations
                    if sum(e_t.values()) <(self.total_agents-1):
                        observation, reward, termination, truncation, info = env.last()
        
                        observation['observation'] =  self.obs_converter(
                                                        torch.tensor(
                                                            observation['observation']
                                                        ).to(self.device,dtype=torch.float32),
                                                        num_classes = self.total_agents+1)
                        
                        episodes[step] = curren_epi
                        obs[step] = observation['observation']#torch.Tensor(observation['observation']).to(self.device) #sould i not add it .... meaning this is the last observation after the player dies
                        action_masks[step] = torch.Tensor(observation['action_mask']).to(self.device)
                        
                        #curr_agent = agent#int(agent[-1])
                        current_agent[step] = curr_agent = agent
                        current_phase[step] = phase = env.phase_selection
                        phase_mapping = self.map_agent_phase_hot(phase,num_classes = self.total_phases).float()
                        
                        curr_agent_mapping = self.map_agent_phase_hot(int(curr_agent)-1,
                                                                      num_classes = self.total_agents 
                                                                     ).float()
                        
                        current_troops_count[step] = torch.Tensor([env.board.agents[i].bucket for i in env.possible_agents]).to(self.device)
                    

                        #model_in = torch.Tensor(torch.hstack((observation['observation'].reshape(-1),
                        #                                      torch.tensor(observation['action_mask'].reshape(-1)).to(self.device)*(curr_agent == self.hero),
                        #                   phase_mapping,
                        #                   curr_agent_mapping,
                        #                   torch.tensor([env.board.agents[ self.hero#curr_agent
                        #                                                ].bucket ]).to(self.device)))[None,:]#.repeat(3,axis = 0)
                        #                        ).float()


                        

                        for i in self.hero_agents_list:
                            self.hero_agents_list[i].model.current_model_in(observation,curr_agent,
                                                                            phase_mapping,curr_agent_mapping,
                                                                            env_board_agents=env.board.agents)
                            self.hero_agents_list[i].model.update_CL_sample_store(curr_agent_=i,
                                                                                  inp = {'step':step_count,'act_2':[] , 'curr_reward_list':[]
                                          },before_action=True)
                            
                        
                        #if e_t[curr_agent]:
                            #print('heeee')
                            
                        if termination or truncation: #this never happens ... the agent is removed from the current agent list and processed after the end of the cycle
                            
                            action = None

                            act = self.actor(torch.Tensor(model_in).to(self.device))
                            #act = self.
                            #act, logprob, _, value = agent_mod.get_action_and_value(model_in)
                            #values[step] = value.flatten() # so even if we are removing the guy ... we need to know what is the action he would 
                                                                #have taken at this point and what would have been its value
                            actions[step] = act #even after going what would have been
                            #logprobs[step] = logprob        
                        else:
                            
                            mask = observation["action_mask"]
                            if (self.global_step < self.args.learning_starts) or (
                                np.random.rand() > min(
                                                ((curren_epi)/((self.num_iterations*self.num_episodes)/10))
                                                , 0.95)
                                                #) or (agent != self.the_hero_agent) 
                                                )or ( agent not in self.hero_agents_list):
        
                                
                                action = env.action_space(agent).sample()
                                #part_0 =np.random.choice(np.where(env.board.calculated_action_mask(agent))[0])
                                part_0 =np.random.choice(np.where(observation['action_mask'])[0])
                                action = torch.Tensor([[[part_0],[np.around(action[1],2)]]]).to(self.device)
                                action = action[:,:,0]
                            else:

                                #need to update this
                                action = self.hero_agents_list[curr_agent].action_predict()
                                
                            
                                #action = self.actor(torch.Tensor(model_in).to(self.device))
                            actions[step] = action
                            curr_agent_ = int(curr_agent)
        
                            if not observation['action_mask'][action[:,0].long()]: 
                                fault_condition =True
                                
                                #self.faulting_player = agent

                                


                                if  curr_agent_ in self.hero_agents_list:
                                    self.hero_agents_list[curr_agent_].bad_move_count+=1
                                    self.hero_agents_list[curr_agent_].bad_move_phase_count[int(current_phase[step][0])]+=1  # when is the where_is_it_performing_bad_really
                                    #print('here',agent, action, observation['action_mask'])
                            
        
                            if  curr_agent_ in self.hero_agents_list:
                                self.hero_agents_list[curr_agent_].move_count[int(current_phase[step][0])]+=1  
                            #if self.the_hero_agent == curr_agent:
                                #move_count[int(current_phase[step][0])]+=1        
        
        
                        #print('here',agent, action)
                        if action != None :
                            act_2 = action.detach().cpu().numpy()[0]#list([action.detach().cpu().numpy()[0][0], max(action.detach().cpu().numpy()[0][1],0.1) ])
                            act_2 = list([act_2[0], max(act_2[1],0.001) ])
                        else:
                            act_2 = action
                            
                        env.step(act_2 if action != None else None)        
        
        
                        if action == None:
                            print('heeee')
                            rewards[step] = np.zeros(self.total_agents) # should i keep it -1? .... hm i dont think so .
                            dones[step] = np.zeros(self.total_agents) # frankly the guys is already done so we really dont have to do anything here.... this is the state post termination for a loser 
                            # but btw this is for the next agent ... action == None means in the last action the previous agent would have been removed.
                            #values[step] = 
                        else:
        
                            
                            curr_reward_list =  env.curr_rewards
                            
                            if (step_count == (self.episode_time_lim-1)): # draw reward
                                is_draw=1
                                curr_reward_list = {i:-100 for i in env.possible_agents }

                            

                            #if self.hero == curr_agent_:
                            #    DT_input['action'][-1]  =act_2
                            #DT_input['return_to_go'][-1] -=  curr_reward_list[self.hero]
                            #returntogo[step] = DT_input['return_to_go'][-1]

                            
                            self.hero_agents_list[curr_agent_].model.update_CL_sample_store(curr_agent_=curr_agent_,
                                          inp = {'step':step_count,'act_2':act_2 , 'curr_reward_list':curr_reward_list
                                          },before_action=False)





                            
                            rewards_2[step] = torch.Tensor([curr_reward_list[i] for i in env.possible_agents]).to(self.device)
                            if step >1:
                                dones_2[step] = torch.Tensor([ int(env.terminations[i]) - dones_2[step-1,i-1]  for i in env.possible_agents]).to(self.device)
                            else:
                                dones_2[step] = torch.Tensor([env.terminations[i] for i in env.possible_agents]).to(self.device)
                                
                            for i in env.possible_agents:
                                if i != curr_agent:
                                    self.gamma_t[i]+=1
                                else:
                                    self.gamma_t[i] =0
        
                                if (step_count == (self.episode_time_lim-1)):
                                    cr_rew = -100
                                    term = True
                                else:
                                    cr_rew = env.curr_rewards[i]
                                    term = env.terminations[i]

                                next_step_ = step-self.gamma_t[i]
                                rewards[next_step_,i-1] += (self.args.gamma**self.gamma_t[i])*cr_rew
                                t_next[next_step_,i-1] = self.gamma_t[i]
                                dones[next_step_,i-1] = torch.Tensor([term]).to(self.device) #so the panetly has to be added but attributions is really difficult
        
                        #list_curr_reward_list = np.array(list(curr_reward_list.values()))
                        
                        #if sum(curr_reward_list.values()) == -300:
                            #print('here')
                            #is_draw=1
        
                        
                        for age_i in env.possible_agents:
                            
                            total_rewards[age_i]+=curr_reward_list[age_i] #env.curr_rewards[age_i] if (step_count != episode_time_lim) else -100
                                    
                        
                        step +=1
                        self.global_step+=1
        
                    else:
                        print('done:',env.terminations,#env.terminations.values(),
                              ",total_reward:",total_rewards, ',iteration:',iteration,",episode:", episode )
                        break    
                
        
        
        
                    step_count+=1
                    
                    if (self.global_step == self.num_steps) :# or (fault_condition and (fa ulting_player != agent) and (len(env.agents)==0)):
                        break
                    elif (step_count == self.episode_time_lim):
                        break
                        
                #print(rewards[step-2])
                if self.global_step == self.num_steps:
                    break 

                for i in self.hero_agents_list:
                    self.hero_agents_list[i].position =self.total_agents
                    
                #[ position = 3 for i in ] 
                for k_,(i_,j_) in enumerate(sorted([(j_,i_) for i_,j_ in total_rewards.items()],reverse=True) 
                      ):
                    if int(j_) in self.hero_agents_list:
                        self.hero_agents_list[int(j_)].position = k_+1
                        
                        
                    #if j_==self.the_hero_agent:
                    #    position = k_+1

                cur_epi_list = (episodes == curren_epi)
                if self.args.TB_log:
                    self.write_exploring(is_draw,#position,
                            curren_epi,step,
                            total_rewards,#bad_move_count
                            #,bad_move_phase_count,
                            #move_count,
                            observation,
                            env,
                            cur_epi_list)

                #paths = []

                for i in self.hero_agents_list:
                    self.hero_agents_list[i].model.update_train_data(
                         step_count,
                         obs,
                            ob_space_shape,
                            rewards_2,
                            dones_2,
                            actions,
                            action_masks,
                            current_agent,
                            current_agent,
                            current_phase,
                            current_troops_count,
                            map_agent_phase_vector = self.map_agent_phase_vector
                         )