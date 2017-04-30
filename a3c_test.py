import time
import pickle
from datetime import date
import json

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from a3c_model import ActorCritic

import gym

from tensorboard_logger import configure, log_value


from loveletter.env import LoveLetterEnv
from loveletter.agents.random import AgentRandom



# api_key = ''
# with open('api_key.json', 'r+') as api_file:
#     api_key = json.load(api_file)['api_key']

evaluation_episodes = 100

def test(rank, args, shared_model, dtype):
    test_ctr = 0
    torch.manual_seed(args.seed + rank)

    #set up logger
    timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    run_name = args.save_name + '_' + timestring
    configure("logs/run_" + run_name, flush_secs=5)


    env = LoveLetterEnv(AgentRandom(args.seed + rank), args.seed + rank)
    env.seed(args.seed + rank)
    state = env.reset()

    model = ActorCritic(state.shape[0], env.action_space).type(dtype)

    model.eval()

    state = torch.from_numpy(state).type(dtype)
    reward_sum = 0
    max_reward = -99999999
    done = True

    start_time = time.time()


    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256).type(dtype), volatile=True)
            hx = Variable(torch.zeros(1, 256).type(dtype), volatile=True)
        else:
            cx = Variable(cx.data.type(dtype), volatile=True)
            hx = Variable(hx.data.type(dtype), volatile=True)

        value, logit, (hx, cx) = model(
            (Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.cpu().numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))

            # if not stuck or args.evaluate:
            log_value('Reward', reward_sum, test_ctr)
            log_value('Episode length', episode_length, test_ctr)

            if reward_sum >= max_reward:
                # pickle.dump(shared_model.state_dict(), open(args.save_name + '_max' + '.p', 'wb'))
                torch.save(shared_model.state_dict(), args.save_name + '_max')
                max_reward = reward_sum

            reward_sum = 0
            episode_length = 0
            state = env.reset()
            test_ctr += 1

            if test_ctr % 10 == 0 and not args.evaluate:
                # pickle.dump(shared_model.state_dict(), open(args.save_name + '.p', 'wb'))
                torch.save(shared_model.state_dict(), args.save_name)
            if not args.evaluate:
                time.sleep(60)
            elif test_ctr == evaluation_episodes:
                # Ensure the environment is closed so we can complete the submission
                env.close()
                # gym.upload('monitor/' + run_name, api_key=api_key)

        state = torch.from_numpy(state).type(dtype)
