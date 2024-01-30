from Env.TetEnv import TetrisEnv
from DDQN.Agent import DDQNAgent
from Utills import un_squeeze
from tqdm import tqdm

if __name__ == "__main__":

    env = TetrisEnv()
    agent = DDQNAgent(gamma=0.99, epsilon=1, batch_size=64, n_actions=5, input_dims=((1, 22, 10), 6), mem_size=500_000, lr=3e-4)
    scores, eps_history, lines = [], [], []

    n_episodes = 10**7

    for i in tqdm(range(n_episodes), desc='Episodes'):
        board, block, reward, done = env.reset()
        cnt = 0
        loss = 0
        r = 0
        while not done:
            action = agent.choose_action([un_squeeze(board), block])
            board_, block_, reward, done = env.step(action)
            agent.store_transition(un_squeeze(board), block, action, reward, un_squeeze(board_), block_, done)
            if agent.memory.mem_cntr >= agent.batch_size:
                loss += agent.learn()
            r += reward
            cnt+=1
            board, block = board_, block_
        tqdm.write(f"Episode {i+1}: Time steps={cnt}, Lines cleared={env.lines-1}, Epsilon={agent.epsilon}, Avg Loss={loss/cnt}, Avg Reward={r/cnt}")


