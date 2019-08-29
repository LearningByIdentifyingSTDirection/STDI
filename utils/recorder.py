import os

class AverageRecorder:
	def __init__(self):
		self.clear()

	def add(self, value):
		self.sum += value
		self.n += 1

	def get(self):
		return self.sum/self.n

	def clear(self):
		self.sum = 0.
		self.n = 0.

class AccuracyRecorder:
	def __init__(self):
		self.clear()

	def add(self, value, n):
		self.sum += value
		self.total += n

	def get(self):
		return self.sum/self.total

	def clear(self):
		self.sum = 0.
		self.total = 0.

class FileRecorder:
	def __init__(self, filename):
		self.filename = filename

	def add(self, items):
		items = map(lambda x:str(x), items)
		if not os.path.exists(self.filename):
			lines = []
		else:
			with open(self.filename, 'r') as f:
				lines = f.readlines()
		line = ','.join(items) + '\n'
		lines.append(line)
		with open(self.filename, 'w') as f:
			f.writelines(lines)
