import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


def animate(i):
    # Read data from csv
    data = pd.read_csv('q_value_per_episode.csv')
    episodes = data['episodes']
    q_values = data['q-values']

    plt.cla()
    plt.title('Learning curve of Q-learning')
    plt.xlabel('Epoch', loc='center')
    plt.ylabel('Q-values', loc='center')
    plt.grid()
    plt.plot(episodes, q_values)


# Plot animated learning curve up to last action
ani = animation.FuncAnimation(plt.gcf(), animate, cache_frame_data=False)
plt.tight_layout()
plt.show()



