from torchvision import transforms

_transform=transforms.Compose([
                       transforms.Resize((224,224)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def transform(image):
    # desired_size = 224
    # old_size = image.size  # old_size[0] is in (width, height) format
    # ratio = float(desired_size)/max(old_size)
    # new_size = tuple([int(x*ratio) for x in old_size])
    # image = image.resize(new_size, Image.Resampling.LANCZOS)
    # new_img = Image.new("RGB", (desired_size, desired_size))
    # new_img.paste(image, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
    return _transform(image)
