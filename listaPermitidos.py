class flabianos:
	""" Lista de invitados"""

	def __init__(self):
		self.Invitados=['antonio','Brad','pedro','carlos']

	def valida_invitado(self,invitado):
		if invitado in self.Invitados:
			print('Bienvenido {}'.format(invitado))
		else:
			print('{}, Detectado en el video'.format(invitado))