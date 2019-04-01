import random


class Cowbull_v2:
	def __init__(self, n_long, n_size):
		self.numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		self.n_long = n_long
		self.n_size = n_size
		self.generated_number = random.sample(self.numbers, self.n_size)
		self.action = [0] * self.n_size
		self.reward_matrix = [[0, 1, 24, 180, 360], [8, 72, 264, 720, 0],
							  [216, 480, 1260, 0, 0], [1440, 0, 0, 0, 0],
							  [5040, 0, 0, 0, 0]]
		self.done = False
		self.reward = 0
		self.result = 'None'

	def step(self, action):
		self.action = action
		if len(self.action) != len(list(set(self.action))):
			return (-100, False)
		else:
			self.reward, self.done, self.result = self.bulls_cows(
			self.action, self.generated_number)
			return self.reward, self.done

	def reset(self):
		self.__init__(self.n_long, self.n_size)
		return

	def render(self):
		print("Last Action:{} \nReward:{} \nResult:{}".format(
			self.action, self.reward, self.result))
		print(self.generated_number)

	def bulls_cows(self, P_1, P_2):
		bulls = 0
		cows = 0
		done = False
		for pos_1, n_1 in enumerate(P_1):
			for pos_2, n_2 in enumerate(P_2):
				if n_1 == n_2 and pos_1 == pos_2:
					bulls += 1
				elif n_1 == n_2:
					cows += 1
		reward = self.reward_matrix[bulls][cows]
		result = '{}B, {}C'.format(bulls, cows)
		if bulls == 4:
			done = True
		return reward, done, result