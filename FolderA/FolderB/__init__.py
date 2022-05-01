from gym.envs.registration import register


# Useful part
register(id='AntLeg2-v0',
         entry_point='FolderB.envs:AntLeg2',
         max_episode_steps=1000,
         reward_threshold=6000.0,
         )

register(id='AntLeg3-v0',
         entry_point='FolderB.envs:AntLeg3',
         max_episode_steps=1000,
         reward_threshold=6000.0,
         )

register(id='Antorigin-v0',
         entry_point='FolderB.envs:Antoriginv0',
         max_episode_steps=1000,
         reward_threshold=6000.0,
         )
register(id='Antnew-v0',
         entry_point='FolderB.envs:Antnewv0',
         max_episode_steps=1000,
         reward_threshold=6000.0,
         )

register(id='Antorigin-v3',
         entry_point='FolderB.envs:Antoriginv3',
         max_episode_steps=1000,
         reward_threshold=6000.0,
         )

register(id='Antnew-v3',
         entry_point='FolderB.envs:Antnewv3',
         max_episode_steps=1000,
         reward_threshold=6000.0,
         )


