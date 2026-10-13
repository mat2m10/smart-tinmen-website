# smart-tinmen-website

The marketing website of **Smart Tinmen**, a studio that builds working
prototypes for real-world robotics, connected devices and ecological
monitoring.

It is a small Rails app with a single page: `PagesController#home` renders
`app/views/pages/home.html.erb`, which is assembled from partials (header,
hero, services, work, process, founders, contact, footer). The founder
profile modal is plain JavaScript loaded through importmap.

## Stack

- Ruby 3.3.0 (`.ruby-version`), Rails 7.2
- Sprockets + importmap-rails (no Node build step), Turbo, Stimulus
- PostgreSQL (required by the Rails config; the site has no models)
- Puma

## Project layout

| Path | What lives there |
| --- | --- |
| `app/views/pages/` | `home.html.erb` and one partial per section |
| `app/assets/stylesheets/home.css` | All page styles |
| `app/javascript/controllers/home.js` | Founder modal open/close logic |
| `app/javascript/controllers/founder_modal.js` | Founder profile data shown in the modal |
| `app/assets/images/` | Founder photos, logo and favicon sizes |
| `notebooks/` | Python/matplotlib notebook and helpers that draw the logo (not used at runtime) |

## Run it locally

Prerequisites: Ruby 3.3.0 and a running PostgreSQL server that your user can
connect to.

```sh
bin/setup              # bundle install, create/prepare the database, start the server
# or, step by step:
bundle install
bin/rails db:prepare
bin/rails server       # http://localhost:3000
```

## Tests and checks

These are the same commands CI runs (`.github/workflows/ci.yml`):

```sh
bin/rails test         # controller tests (needs PostgreSQL)
bin/rubocop            # style (rubocop-rails-omakase)
bin/brakeman --no-pager
bin/importmap audit
```

## Editing content

- Text for each section: the matching partial in `app/views/pages/`.
- Founder cards: `app/views/pages/_founders.html.erb`; the full profile shown
  in the modal: `app/javascript/controllers/founder_modal.js`. The key in
  `founders` must match the card's `data-founder` attribute.
- Page title, description and link-preview (Open Graph) tags:
  `app/views/layouts/application.html.erb`.

## Logo and favicons

The logo is drawn in Python in `notebooks/draw_face.ipynb` (helpers in
`notebooks/draw_face_helpers/`). Its export cell writes every favicon and
logo size straight into `app/assets/images/`. Re-run it after changing the
design, then commit the regenerated images.

## Deployment

TODO: document the production host and deploy steps.

What the repository shows: there is a production `Dockerfile` (Rails
default, "use with Kamal or build'n'run by hand"), but no Kamal
`config/deploy.yml`; an early commit is titled "Ensure pg and puma for
Heroku", but there is no `Procfile` or `app.json`. Production needs
`RAILS_MASTER_KEY` (from `config/master.key`, which is not in git) and a
PostgreSQL database (`DATABASE_URL`, or
`SMART_TINMEN_WEBSITE_DATABASE_PASSWORD` with the names in
`config/database.yml`).

## License

TODO: choose a license. Until then, no license is granted: the code and the
Smart Tinmen name, logo and content are © Smart Tinmen, all rights reserved.
