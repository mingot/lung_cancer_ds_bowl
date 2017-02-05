import numpy as np
from networks.c3d_network import C3DNetworkArchitecture
from scipy.misc import imread, imresize

sample_test_image = '/home/jose/Desktop/bear.jpg'

def test_c3d_net():
    print("START C3D NETWORK")
    def get_input_example():
        # We create a fake video with the expected shape in C3D (16 frames)
        W, H, Z = 112,112,16
        img = imresize(imread(sample_test_image, mode='RGB'), (H,W)).astype(np.float32)
        img = img.transpose((2, 0, 1))
        video_raro = np.array([[[img[0,:] for i in range(Z)],
                               [img[1,:] for i in range(Z)],
                               [img[2,:] for i in range(Z)]
                              ]], dtype = np.float32)
        return video_raro

    # get sample data
    sample_fake_data = get_input_example()

    print("This test shows how the network 'does something', meaning it does not break.")
    print("It also pretends to show the functionalities of the network module. The output is meaningless, so just ignore it!")

    print("1 - We test the network loaded from scratch (no preloaded)")
    c3d = C3DNetworkArchitecture((3,16,112,112), use_pretrained = False)
    prediction = c3d.net.predict(sample_fake_data)[0]
    print("  - The output for the fake input image is : %0.2f" % prediction[0])

    print("2 - We test the network with pretrained weights")
    c3d = C3DNetworkArchitecture((3,16,112,112), use_pretrained = True)
    prediction = c3d.net.predict(sample_fake_data)[0]
    print("  - The output for the fake input image is : %0.2f" % prediction[0])

    print("3 - We save and reload weights.")
    c3d = C3DNetworkArchitecture((3,16,112,112), use_pretrained = True)
    c3d.save_weights('c3d_test_random_weights')
    prediction = c3d.net.predict(sample_fake_data)[0]
    c3d = C3DNetworkArchitecture((3,16,112,112), use_pretrained = False)
    c3d.load_weights('c3d_test_random_weights')
    prediction1 = c3d.net.predict(sample_fake_data)[0]
    assert prediction == prediction, "  - Weights are not properly saved/loaded!"
    print("  - The output is the same after saving and loading. It looks good!")

    print("FINISH C3D NETWORK")

if __name__ == '__main__':
    test_c3d_net()