import spectrum
from spectrum import SpectrumEnv
from spectrum import Suit
# env = gym.make('Spectrum')

def main():
    print("Running")
    env = SpectrumEnv()
    observation = env.reset()
    for t in range(100):
        env.render('human')
        print("Printing observation:")
        print(observation)
        action = env.action_space.sample()
        print("Action:", action)
        observation, reward, done, info = env.step(action)
        if done:
            print ("Episode finished after {} timesteps".format(t+1))
            break

if __name__ == '__main__':
    main()
