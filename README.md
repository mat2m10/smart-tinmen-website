# smart-tinmen-website

The marketing website of **Smart Tinmen**, a studio that builds working
prototypes for real-world robotics, connected devices and ecological
monitoring. Live at [smarttinmen.com](https://smarttinmen.com).

It is a small Rails app with a single page: `PagesController#home` renders
`app/views/pages/home.html.erb`, which is assembled from one partial per
section. There are no models.

## Stack

- Ruby 3.3.12 (`.ruby-version`), Rails 8.1
- Sprockets + importmap-rails (no Node build step), Turbo, Stimulus
- PostgreSQL (`pg` gem; required by the Rails config even though the site has no models)
- Puma
- Hosted on Heroku (app `smart-tinmen-website`, stack heroku-24)

## Project layout

| Path | What lives there |
| --- | --- |
| `app/views/pages/` | `home.html.erb` and one partial per section (header, hero, services, work, record, process, team, contact, footer), plus `sitemap.xml.erb` |
| `app/assets/stylesheets/home.css` | All page styles |
| `app/javascript/controllers/` | Stimulus controllers (`clipboard_controller.js` copies the contact email) |
| `app/assets/images/logo.svg` | **Source of the logo.** Every logo, favicon, apple-touch and android PNG in `app/assets/images/` is rendered from it |
| `design/og-image.html` | **Source of the link-preview image** `app/assets/images/og-image.png` (1200x630) |
| `app/views/layouts/application.html.erb` | Page title, meta description, Open Graph / Twitter tags |
| `notebooks/` | Legacy Python notebook that drew the previous logo. Not used at runtime, and superseded by `logo.svg` |

## Run it locally

Prerequisites: Ruby 3.3.12 and a running PostgreSQL server that your user can
connect to.

```sh
bin/setup      # bundle install, prepare the database, then start the server
# or, step by step:
bundle install
bin/rails db:prepare
bin/dev        # http://localhost:3000
```

## Tests and checks

These are the commands CI runs (`.github/workflows/ci.yml`) on pushes to
`master` and on pull requests:

```sh
bin/rails db:test:prepare test test:system   # needs PostgreSQL (and Chrome for system tests)
bin/rubocop                                  # style (rubocop-rails-omakase)
bin/brakeman --no-pager                      # security static analysis
bin/importmap audit                          # JS dependency vulnerabilities
```

## Editing content

- Text for each section: the matching partial in `app/views/pages/`.
- Styles: `app/assets/stylesheets/home.css`.
- Title, description and link-preview tags: `app/views/layouts/application.html.erb`.
- Sitemap: `app/views/pages/sitemap.xml.erb` (served at `/sitemap.xml`).

## Logo and link-preview image

- **Logo:** edit `app/assets/images/logo.svg`, then re-render the PNG sizes
  (logo, `favicon-*`, `apple-touch-icon-180`, `android-*`) from it at their
  existing pixel sizes and commit them. The 16–48 px favicons use a simplified
  variant (no rivets, thicker lines) so they stay legible.
- **Link preview:** edit `design/og-image.html`, render it at 1200x630 in a
  headless browser and save the screenshot over `app/assets/images/og-image.png`.

## Deploy

Deploys are **manual**: Heroku does not auto-deploy from GitHub. After a
change is merged to `master` and CI is green:

```sh
git push heroku master:main
```

The `heroku` git remote is `https://git.heroku.com/smart-tinmen-website.git`.
Until you run that push, the live site keeps serving the previous release.

## License

TODO: choose a license. Until then, no license is granted: the code and the
Smart Tinmen name, logo and content are © Smart Tinmen, all rights reserved.
