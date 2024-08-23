class Trainer:
    """
    ## Trainer
    """

    def __init__(self, *,Args
                 ):
        # #### Configurations

        self.args = Args()#tyro.cli(Args)
        self.args.batch_size = int(self.args.num_envs * self.args.num_steps)
        self.args.minibatch_size = int(self.args.batch_size // self.args.num_minibatches)
        self.args.num_iterations = self.args.total_timesteps // self.args.batch_size
        self.run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        

        TB_log = True
        if TB_log:    
            writer = SummaryWriter(f"runs/{run_name}")
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )
        
        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        
        
        self.playe_r = 1#"agent_1" #
        
        self.num_steps = 1000000
        self.action_shape = (2,)

        self.env_config = dict(render_mode = 'rgb_array', default_attack_all  = True,
                render_ = True,agent_count  = 3,use_placement_perc=True,render_=False)

        self.env = env_risk(**env_config)
        
        self.env.reset(seed=42)



        
        sample_obs = self.obs_converter(torch.tensor(self.env.last()[0]['observation']),num_classes = 4)
        
        self.ob_space_shape = sample_obs.shape #env.observation_space(playe_r)['observation'].shape
        self.action_mask_shape = env.observation_space(playe_r)['action_mask'].shape
        self.total_agents  = len(env.possible_agents)
        self.tota_phases = len(env.phases)
        #self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        
        
        
        
        
        self.agent_list = list(env.possible_agents)
        self.global_step = 0
        
        
        self.faulting_player = ""
        
        self.num_episodes = 100
        
        self.phase = 0
        self.args.minibatch_size = 128        


        
        self.qnet_config_dict = dict(action_space = self.env.action_space(self.playe_r
                                                                         ).shape[0],
                                    ob_space=np.prod(self.ob_space_shape
                                                    )+np.prod(self.action_mask_shape)
                                         +1*( self.total_agents -1) #the current_agent +1#who actor agent was
                                         +1*(self.tota_phases -1)#the current phase
                                         +1 # the number of troops
                               )
        self.actor_config_dict =  dict(env=self.env,
                        action_space = env.observation_space(self.playe_r)['action_mask'].shape[0],
                        ob_space=np.prod(self.ob_space_shape)+np.prod(self.action_mask_shape)
                                         +1*( self.total_agents-1) #the current_agent +1#who actor agent was
                                         +1*(self.tota_phases -1)#the current phase
                                         +1 # the number of troops
                               )
        
        
        
        
        self.actor = Actor_ddqn(**actor_config_dict).to(device)
        self.qf1 = QNetwork(**qnet_config_dict).to(device)
        self.qf1_target = QNetwork(**qnet_config_dict).to(device)
        self.target_actor = Actor_ddqn(**actor_config_dict).to(device)
        
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()), lr=self.args.learning_rate)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.args.learning_rate)
        
        self.start_time = time.time()
        

        

    def obs_converter(self,  data, num_classes = 4, col =0 ):

        if col != None:
            return torch.concat((nn.functional.one_hot(data[:,col].detach().long(), 
                                                        num_classes = num_classes),
                                      data[:,~col,None]
                                ),axis=1
                               )[:,1:].to(self.device)
    
    def map_agent_phase_hot(self, data,num_classes = 3):
        return nn.functional.one_hot(torch.tensor(data),num_classes = num_classes)[1:].to(self.device)
    
    def map_agent_phase_vector(self, data,num_classes = 3):
        return nn.functional.one_hot(data[:,0].long(), 
                                                            num_classes = num_classes)[:,1:].to(self.device)

    def run_training_loop(self):
        """
        ### Run training loop
        """

        # last 100 episode information
        #tracker.set_queue('reward', 100, True)
        #tracker.set_queue('length', 100, True)


        obs = torch.zeros((self.num_steps,) + self.ob_space_shape).to(self.device)
        actions = torch.zeros((self.num_steps, ) + self.action_shape).to(self.device)
        action_masks = torch.zeros((self.num_steps, ) + self.action_mask_shape).to(self.device)
        current_agent = torch.ones((self.num_steps,1)).to(self.device)*0#-1
        current_phase = torch.zeros((self.num_steps,1)).to(self.device)
        current_troops_count = torch.zeros((self.num_steps,self.total_agents)).to(self.device)
        logprobs = torch.zeros((self.num_steps, )).to(self.device)
        rewards = torch.zeros((self.num_steps, self.total_agents)).to(self.device)
        rewards_2 = torch.zeros((self.num_steps, self.total_agents)).to(self.device)
        dones = torch.zeros((self.num_steps, self.total_agents)).to(self.device)
        values = torch.zeros((self.num_steps,  )).to(self.device)
        episodes = torch.ones((self.num_steps, )).to(self.device)*-1
        t_next = torch.zeros((self.num_steps, self.total_agents)).to(self.device)


        rb = ReplayBuffer(
                args.buffer_size,
                Box(low =0, high=2000, shape =(self.qnet_config_dict['ob_space']+1,), dtype=np.float32),
                Box(low =0, high=2000, shape =(2,), dtype=np.float32),
                self.device,
                handle_timeout_termination=False,
            )



        the_hero_agent = 1

        num_episodes =5
        env = env_risk(**(self.env_config | {"render_mode" : None, "bad_mov_penalization" : 0.01,"render_":False}))
        
        self.args.gamma = gam = 0.99
        gamma_t = {i:0 for i in env.possible_agents}
        
        env.reset(42)
        
        num_iterations = 1000
        episode_time_lim = 5000
        draw_count = 0
        draw_territory_count = 0
        first_count = 0
        second_count = 0
        third_count = 0
        third_count_draw = 0

        
        start_time = time.time()
        for iteration in range(1, num_iterations):
            
            rb = self.sample(

                                obs
                                ,actions
                                ,action_masks
                                ,current_agent
                                ,current_phase
                                ,current_troops_count
                                ,logprobs
                                ,rewards
                                ,rewards_2
                                ,dones
                                ,values
                                ,episodes
                                ,t_next,
                                rb
                 
                            )
            
            self.train(samples)
            


    def sample(self,
              
                    obs
                    ,actions
                    ,action_masks
                    ,current_agent
                    ,current_phase
                    ,current_troops_count
                    ,logprobs
                    ,rewards
                    ,rewards_2
                    ,dones
                    ,values
                    ,episodes
                    ,t_next,
                    rb              
              ):
        """
        ### Sample data with current policy
        """

        #rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)

        with torch.no_grad():
            # sample `worker_steps` from each worker
            #there are no worker steps... rather there are full episodes

            step = 0
            fault_condition = False
            clear_output(wait=True)
        
        
            
            for episode in range(num_episodes):#num_episodes):
                
                total_rewards = {i:0 for i in env.possible_agents} #i can report this
                action=1
                if fault_condition:
                    env = env_risk(**(self.env_config | {"render_mode" : None,"bad_mov_penalization" : 0.01,"render_":False})
                                      )#game.env(render_mode=None)

                curren_epi = episode + (iteration-1)*num_episodes
                
                env.reset(curren_epi) #for riplication
                
                fault_condition = False
                step_count = 0
                bad_move_count = 0
                bad_move_phase_count = {i:0 for i in env.phases}
                move_count =  {i:0 for i in env.phases}
                is_draw = 0
                draw_territory_count = 0
                is_third = 0

                for agent in env.agent_iter():
                    e_t = env.terminations
                    if sum(e_t.values()) <2:
                        observation, reward, termination, truncation, info = env.last()
        
                        observation['observation'] =  self.obs_converter(
                                                        torch.tensor(
                                                            observation['observation']
                                                        ).to(device,dtype=np.float32))
                        
                        episodes[step] = episode + (iteration-1)*num_episodes
                        current_phase[step] = phase            


                        obs[step] = observation['observation']#torch.Tensor(observation['observation']).to(device) #sould i not add it .... meaning this is the last observation after the player dies
                        action_masks[step] = torch.Tensor(observation['action_mask']).to(device)
                        curr_agent = agent#int(agent[-1])
                        current_agent[step] = curr_agent
                        current_phase[step] = env.phase_selection
                        phase_mapping = map_agent_phase_hot(env.phase_selection,num_classes = len(env.phases)).float()
                        curr_agent_mapping = map_agent_phase_hot(int(curr_agent)-1,num_classes = len(env.possible_agents)).float()
                        
                        current_troops_count[step] = torch.Tensor([env.board.agents[i].bucket for i in env.possible_agents]).to(device)
                    

                        model_in = torch.Tensor(torch.hstack((observation['observation'].reshape(-1),torch.tensor(observation['action_mask'].reshape(-1)).to(device),
                                           phase_mapping,
                                            curr_agent_mapping,
                                           torch.tensor([env.board.agents[curr_agent].bucket ]).to(device)))[None,:]#.repeat(3,axis = 0)
                                                ).float()
                
                        
                        if termination or truncation:
                            action = None
                            
                            act, logprob, _, value = agent_mod.get_action_and_value(model_in)
                            values[step] = value.flatten() # so even if we are removing the guy ... we need to know what is the action he would 
                                                                #have taken at this point and what would have been its value
                            actions[step] = act #even after going what would have been
                            logprobs[step] = logprob        
                        else:
                            mask = observation["action_mask"]
                            if (global_step < args.learning_starts) or (
                        np.random.rand() > min(((episode + (iteration-1)*num_episodes)/((num_iterations*num_episodes)/10))
                            , 0.95)) or (agent != the_hero_agent):
        
                                
                                action = env.action_space(agent).sample()
                                part_0 =np.random.choice(np.where(env.board.calculated_action_mask(agent))[0])
                                action = torch.Tensor([[[part_0],[np.around(action[1],2)]]]).to(device)
                                action = action[:,:,0]
                            else:
                                
                                action = actor(torch.Tensor(model_in).to(device))
                            actions[step] = action
        
                            if not observation['action_mask'][action[:,0].long()]: 
                                fault_condition =True
                                faulting_player = agent
        
                                if the_hero_agent == curr_agent:
                                    bad_move_count+=1
                                    bad_move_phase_count[int(current_phase[step][0])]+=1  # when is the where_is_it_performing_bad_really
                                    #print('here',agent, action, observation['action_mask'])
                            
        
                            if the_hero_agent == curr_agent:
                                move_count[int(current_phase[step][0])]+=1

            
            for t in range(self.worker_steps):
                # `self.obs` keeps track of the last observation from each worker,
                #  which is the input for the model to sample the next action
                obs[:, t] = self.obs
                # sample actions from $\pi_{\theta_{OLD}}$ for each worker;
                #  this returns arrays of size `n_workers`
                pi, v = self.model(obs_to_torch(self.obs))
                values[:, t] = v.cpu().numpy()
                a = pi.sample()
                actions[:, t] = a.cpu().numpy()
                log_pis[:, t] = pi.log_prob(a).cpu().numpy()

                # run sampled actions on each worker
                for w, worker in enumerate(self.workers):
                    worker.child.send(("step", actions[w, t]))

                for w, worker in enumerate(self.workers):
                    # get results after executing the actions
                    self.obs[w], rewards[w, t], done[w, t], info = worker.child.recv()

                    # collect episode info, which is available if an episode finished;
                    #  this includes total reward and length of the episode -
                    #  look at `Game` to see how it works.
                    if info:
                        tracker.add('reward', info['reward'])
                        tracker.add('length', info['length'])

            # Get value of after the final step
            _, v = self.model(obs_to_torch(self.obs))
            values[:, self.worker_steps] = v.cpu().numpy()

        # calculate advantages
        advantages = self.gae(done, rewards, values)

        #
        samples = {
            'obs': obs,
            'actions': actions,
            'values': values[:, :-1],
            'log_pis': log_pis,
            'advantages': advantages
        }

        # samples are currently in `[workers, time_step]` table,
        # we should flatten it for training
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs':
                samples_flat[k] = obs_to_torch(v)
            else:
                samples_flat[k] = torch.tensor(v, device=device)

        return samples_flat

    def train(self, samples: Dict[str, torch.Tensor]):
        """
        ### Train the model based on samples
        """

        # It learns faster with a higher number of epochs,
        #  but becomes a little unstable; that is,
        #  the average episode reward does not monotonically increase
        #  over time.
        # May be reducing the clipping range might solve it.
        for _ in range(self.epochs()):
            # shuffle for each epoch
            indexes = torch.randperm(self.batch_size)

            # for each mini batch
            for start in range(0, self.batch_size, self.mini_batch_size):
                # get mini batch
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start: end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

                # train
                loss = self._calc_loss(mini_batch)

                # Set learning rate
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.learning_rate()
                # Zero out the previously calculated gradients
                self.optimizer.zero_grad()
                # Calculate gradients
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                # Update parameters based on gradients
                self.optimizer.step()

    @staticmethod
    def _normalize(adv: torch.Tensor):
        """#### Normalize advantage function"""
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def _calc_loss(self, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        ### Calculate total loss
        """

        # $R_t$ returns sampled from $\pi_{\theta_{OLD}}$
        sampled_return = samples['values'] + samples['advantages']

        # $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$,
        # where $\hat{A_t}$ is advantages sampled from $\pi_{\theta_{OLD}}$.
        # Refer to sampling function in [Main class](#main) below
        #  for the calculation of $\hat{A}_t$.
        sampled_normalized_advantage = self._normalize(samples['advantages'])

        # Sampled observations are fed into the model to get $\pi_\theta(a_t|s_t)$ and $V^{\pi_\theta}(s_t)$;
        #  we are treating observations as state
        pi, value = self.model(samples['obs'])

        # $-\log \pi_\theta (a_t|s_t)$, $a_t$ are actions sampled from $\pi_{\theta_{OLD}}$
        log_pi = pi.log_prob(samples['actions'])

        # Calculate policy loss
        policy_loss = self.ppo_loss(log_pi, samples['log_pis'], sampled_normalized_advantage, self.clip_range())

        # Calculate Entropy Bonus
        #
        # $\mathcal{L}^{EB}(\theta) =
        #  \mathbb{E}\Bigl[ S\bigl[\pi_\theta\bigr] (s_t) \Bigr]$
        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        # Calculate value function loss
        value_loss = self.value_loss(value, samples['values'], sampled_return, self.clip_range())

        # $\mathcal{L}^{CLIP+VF+EB} (\theta) =
        #  \mathcal{L}^{CLIP} (\theta) +
        #  c_1 \mathcal{L}^{VF} (\theta) - c_2 \mathcal{L}^{EB}(\theta)$
        loss = (policy_loss
                + self.value_loss_coef() * value_loss
                - self.entropy_bonus_coef() * entropy_bonus)

        # for monitoring
        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()

        # Add to tracker
        tracker.add({'policy_reward': -policy_loss,
                     'value_loss': value_loss,
                     'entropy_bonus': entropy_bonus,
                     'kl_div': approx_kl_divergence,
                     'clip_fraction': self.ppo_loss.clip_fraction})

        return loss


    def destroy(self):
        """
        ### Destroy
        Stop the workers
        """
        for worker in self.workers:
            worker.child.send(("close", None))
