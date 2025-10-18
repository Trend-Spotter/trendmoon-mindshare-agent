# Mindshare UI

React application for Mindshare UI.
Served by the Mindshare agent, designed to be consumed by the agent and available in [Pearl](https://github.com/valory-xyz/olas-operate-app).

## 🚀 Development

1.  Install via `yarn install`
2.  Run via `npx nx serve mindshare-ui`
    -   The app will be available at `http://localhost:4200`
3.  Build for production via `npx nx build mindshare-ui`
    -   The build will be available in the `dist/apps/mindshare-ui` directory
    -   `/build` is the output directory, and can be served statically

## 🧪 Mock Data

To mock, update the `IS_MOCK_ENABLED` in `config.ts` to `true` and the app will use the mock data instead of the API. To enable the chat mock, set `isChatEnabled` in `mockFeatures.ts` to `true` as well.

## 📦 Release process

1.  Bump the version in `package.json`
2.  Push a new tag to the repository
3.  The CI will build and release the contents of the `dist/apps/mindshare-ui` directory to a zip file.
