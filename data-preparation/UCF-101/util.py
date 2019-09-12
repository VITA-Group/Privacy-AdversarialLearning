import numpy as np

def gaussian_kernel(sigma, ksize):
    radius = (ksize - 1) / 2.0
    x, y = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    sigma = sigma ** 2
    k = 2 * np.exp(-0.5 * (x ** 2 + y ** 2) / sigma)
    k = k / np.sum(k)
    return k

def tile_and_reflect(input):
    tiled_input = np.tile(input, (3, 3))
    rows = input.shape[0]
    cols = input.shape[1]

    for i in range(3):
        tiled_input[i * rows:(i + 1) * rows, 0:cols] = np.fliplr(tiled_input[i * rows:(i + 1) * rows, 0:cols])
        tiled_input[i * rows:(i + 1) * rows, -cols:] = np.fliplr(tiled_input[i * rows:(i + 1) * rows, -cols:])

    for i in range(3):
        tiled_input[0:rows, i * cols:(i + 1) * cols] = np.flipud(tiled_input[0:rows, i * cols:(i + 1) * cols])
        tiled_input[-rows:, i * cols:(i + 1) * cols] = np.flipud(tiled_input[-rows:, i * cols:(i + 1) * cols])

    assert (np.array_equal(input, tiled_input[rows:2 * rows, cols:2 * cols]))

    assert (np.array_equal(input[0, :], tiled_input[rows - 1, cols:2 * cols]))
    assert (np.array_equal(input[:, -1], tiled_input[rows:2 * rows, 2 * cols]))
    assert (np.array_equal(input[-1, :], tiled_input[2 * rows, cols:2 * cols]))
    assert (np.array_equal(input[:, 0], tiled_input[rows:2 * rows, cols - 1]))

    return tiled_input

def convolve(input, weights):
    assert (len(input.shape) == 2)
    assert (len(weights.shape) == 2)

    assert (weights.shape[0] < input.shape[0] + 1)
    assert (weights.shape[0] < input.shape[1] + 1)

    output = np.copy(input)
    tiled_input = tile_and_reflect(input)

    rows = input.shape[0]
    cols = input.shape[1]
    hw_row = weights.shape[0] // 2
    hw_col = weights.shape[1] // 2

    for i, io in zip(range(rows, rows * 2), range(rows)):
        for j, jo in zip(range(cols, cols * 2), range(cols)):
            average = 0.0
            overlapping = tiled_input[i - hw_row:i + hw_row,
                          j - hw_col:j + hw_col]
            assert (overlapping.shape == weights.shape)
            tmp_weights = weights
            merged = tmp_weights[:] * overlapping
            average = np.sum(merged)
            output[io, jo] = average
    return output

def gaussian_blur(img, ksize, sigma):
    k = gaussian_kernel(sigma, ksize)
    blurred_img = convolve(img, k)
    return blurred_img

def dsampleWithBlur(image, sigma, ksize, color=True):
    k = gaussian_kernel(sigma, ksize)
    if color:
        ndim = 3
    else:
        ndim = 1
    ds = []
    for i in range(ndim):
        img = image[:, :, i]
        blocks = extract_blocks(img.reshape(240, 320), (ksize, ksize))
        lst = []
        for block in blocks:
            lst.append(np.sum(np.multiply(block, k)))
        ds.append(np.array(lst).reshape(int(img.shape[0] / ksize), int(img.shape[1] / ksize)))
    return np.transpose(np.array(ds), (1, 2, 0))

def dsampleWithAvg(image, ksize, color=True):
    if color:
        ndim = 3
    else:
        ndim = 1
    ds = []
    for i in range(ndim):
        img = image[:, :, i]
        blocks = extract_blocks(img.reshape(240, 320), (ksize, ksize))
        lst = []
        for block in blocks:
            lst.append(np.mean(block))
        ds.append(np.array(lst).reshape(int(img.shape[0] / ksize), int(img.shape[1] / ksize)))
    return np.transpose(np.array(ds), (1, 2, 0))

def extract_blocks(img, blocksize):
    M, N = img.shape
    b0, b1 = blocksize
    return img.reshape(M // b0, b0, N // b1, b1).swapaxes(1, 2).reshape(-1, b0, b1)