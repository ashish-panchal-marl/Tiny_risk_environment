# parameters to Train


exp_name = 'exp12_ddqn_4_agents_2_hero_norm_small',


- learning_rate = 0.0003
- batch_size = 2,
- gamma = 0.99,
num_steps=120000,
num_iterations = 500,
episode_time_lim = 10000,
hero_agent_count = 1,#2,
model_name={1:"transformer_model"}#,2:'transformer_model'}
,entropy=True,
return_prob=2,
actor_wt = 0.5,
CE_wt = 0.01,
small = True,
num_episodes = 1,#5,
context_len = 256,
rtg_scale=1,
shuffle=True,
pin_memory=False,#False,
drop_last=True,
TB_log=False,
learning_starts =100,
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),

pin_memory_device= ("cuda" if torch.cuda.is_available() else "cpu"),

    
env_config = dict(render_mode = None,#'rgb_array', 
                    default_attack_all  = True,
                    agent_count  = 3#4
                    ,use_placement_perc=True,
                    render_=False,
                    bad_mov_penalization = 0.01
                 )
,model_config = dict(
                    n_blocks      =   3,
                    embed_dim     =   128,#128 ,
                    context_len   =   256,#256  ,
                    n_heads       =   1,
                    dropout_p     =   0.1,
                    wt_decay      =   0.0001,
                    warmup_steps  =   100   ,
                    tau           =   0.95,
                    chunk_size    =   128,#64
                    chunk_overlap =   1
                    )

)