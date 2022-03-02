use image::{GenericImageView, Pixel};

fn main()  {

    let img = image::open("sample/test.png").unwrap();
    let (width, height) = img.dimensions();

    let mut raw_img = img.into_rgba8();
    for x in 1..width {
        for y in 1..height {
            let old_pixel = raw_img.get_pixel_mut(x, y);
            old_pixel.apply(|t| -> u8 {
                t / 32 * 32
            })
        }
    }
    raw_img.save("output/test.png").unwrap();
}

