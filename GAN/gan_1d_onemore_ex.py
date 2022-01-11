from numpy import zeros
from numpy import ones
from numpy.random import rand 
from numpy.random import randn
from numpy import hstack
from matplotlib import pyplot

def generate_samples(n=100):
    X1 = rand(n)-0.5
    X2 = X1**3
    X1 = X1.reshape(n,1)
    X2 = X2.reshape(n,1)
    return hstack((X1,X2))

data = generate_samples()
pyplot.scatter(data[:,0],data[:,1])
pyplot.show()

#define standalone discriminator model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

def define_discriminator(n_inputs=2):
    """it is the discriminator model used in gan"""
    model = Sequential()
    model.add(Dense(25,activation='relu',kernel_initializer='he_uniform',input_dim=n_inputs))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def generate_real_samples(n):
    """function to be used to generated real samples"""
    X1 = rand(n)-0.5
    X2 = X1**3
    X1 = X1.reshape(n,1)
    X2 = X2.reshape(n,1)
    X = hstack((X1,X2))
    y = ones((n,1))
    return X,y

def generate_fake_samples(n):
    """used to generate fake samples"""
    X1 = -1+rand(n)*2
    X2 = -1+rand(n)*2
    X1 = X1.reshape(n,1)
    X2 = X2.reshape(n,1)
    X = hstack((X1,X2))
    y = zeros((n,1))
    return X,y

def train_discriminator(model,n_epochs=1000,n_batches=128):
    """It is used to train the discriminator model"""
    half_batch = int(n_batches/2)
    for i in range(n_epochs):
        X_real, y_real = generate_real_samples(half_batch)
        model.train_on_batch(X_real,y_real)
        X_fake, y_fake = generate_fake_samples(half_batch)
        model.train_on_batch(X_fake,y_fake)
        #evaluate the model
        _,acc_real = model.evaluate(X_real,y_real,verbose=0)
        _,acc_fake = model.evaluate(X_fake,y_fake,verbose=0)
        print(i,acc_real,acc_fake)

# model = define_discriminator()
# train_discriminator(model)

# define the standalone generator model
def define_generator(latent_dim, n_outputs=2):
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(n_outputs, activation='linear'))
    return model

def generate_latent_points(latent_dim, n):
    x_input = randn(latent_dim*n)
    x_input = x_input.reshape(n,latent_dim)
    return x_input

def generate_fake_samples(generator, latent_dim, n):
    x_input = generate_latent_points(latent_dim,n)
    X = generator.predict(x_input)
    pyplot.scatter(X[:,0],X[:,1])
    pyplot.show()


latent_dim = 5
model = define_generator(latent_dim)
generate_fake_samples(model,latent_dim,100)

#define the combined gan model
def define_gan(generator, discriminator):
    discriminator.trainable=False 
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator,discriminator)
gan_model.summary()
plot_model(gan_model,to_file='gan_1d_onemore_ex.png',show_shapes=True,show_layer_names=True)

def generate_fake_samples(generator, latent_dim, n):
    x_input = generate_latent_points(latent_dim,n)
    X = generator.predict(x_input)
    y = zeros((n,1))
    return X,y

def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
    x_real, y_real = generate_real_samples(n)
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    X_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    _, acc_fake = discriminator.evaluate(X_fake, y_fake, verbose=0)
    print(epoch, acc_real, acc_fake)
    pyplot.scatter(x_real[:,0],x_real[:,1],color='red')
    pyplot.scatter(X_fake[:,0],X_fake[:,1],color='blue')
    filename = 'generated_plot_onemore_ex_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()    

def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batches=128,n_eval=2000):
    half_batch = int(n_batches/2) 
    for i in range(n_epochs):
        x_real, y_real = generate_real_samples(half_batch)
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_model.train_on_batch(x_real,y_real)
        d_model.train_on_batch(x_fake,y_fake)
        x_gan = generate_latent_points(latent_dim, n_batches)
        y_gan = ones((n_batches,1))
        gan_model.train_on_batch(x_gan,y_gan)

        if((i+1)%n_eval==0):
            summarize_performance(i, g_model, d_model, latent_dim)

train(generator,discriminator,gan_model,latent_dim)

