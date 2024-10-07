#self.hero_agents_list[i].model.update_CL_sample_store(curr_agent_=i,




#episode_length


#for each episode



sum()/














def update_CL_sample_store(
        self,
        curr_agent_,
        inp={'step': None, 'act_2_1': [],'act_2_2': [], 'curr_reward_list': []},
        before_action=True,
        ):

        if before_action == 1 :

            if inp['step'] == 0:

                # print(self.model_in.shape)
                # print(self.model_in.repeat(self.context_len).shape)

                self.DT_input['state'] = self.model_in.repeat(self.context_len,
                        1).to(self.device)[None, :]
            else:

            # if step<self.context_len:

            #    trace[step] = model_in

                self.DT_input['state'][0, 0:-1] = self.DT_input['state'
                        ][0, 1:].clone()
                self.DT_input['state'][0, -1] = self.model_in
                self.DT_input['timestep'][0, 0:-1] = self.DT_input['timestep'][0, 1:].clone()
                self.DT_input['timestep'][0, -1] = inp['step']
                self.DT_input['action_1'][0, 0:-1] = self.DT_input['action_1'][0, 1:].clone()
                self.DT_input['action_2'][0, 0:-1] = self.DT_input['action_2'][0, 1:].clone()
                
                self.DT_input['return_to_go'][0, 0:-1] = self.DT_input['return_to_go'][0, 1:].clone()
        elif before_action == 2 :
            if self.hero == curr_agent_:
                self.DT_input['action_1'][0, -1] = inp['act_2_1']
            else:
                self.DT_input['action_1'][0, -1] = 0

        elif before_action == 3:
            if self.hero == curr_agent_:
                self.DT_input['action_2'][0, -1] = inp['act_2_2']
            else:
                self.DT_input['action_2'][0, -1] = 0
                
        else:
            
                
            self.DT_input['return_to_go'][0, -1] = self.DT_input['return_to_go'][0, -1] -    inp['curr_reward_list']  # [self.hero]
            self.returntogo[inp['step']] = self.DT_input['return_to_go'
                    ][0, -1]



    def action_predict(self, save_R=True, return_R=False,shift=1):
        (#s, 
         a_1,a_2, R) = self.model(timesteps=self.DT_input['timestep'],
                               states=self.DT_input['state'],
                               actions_1=self.DT_input['action_1'],
                               actions_2=self.DT_input['action_2'],
                               returns_to_go=self.DT_input['return_to_go'
                               ][:, :, None])

        
        action_1 = a[0, -1, :].argmax()[None]
        action_2 = a[0,-1, 0][None]

        
        if save_R:
            self.returntogo_pred[self.DT_input['timestep'][0, -1]] =                 R[0, -1]  # R

        if return_R:
            return (action_1, action_2, R[0, -1])
        else:
            return action_1, action_2
